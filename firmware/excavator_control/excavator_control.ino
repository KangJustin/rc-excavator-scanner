/*
 * excavator_control.ino  --  ESP32 bridge: USB serial -> digipots + button MOSFETs
 *
 * Replaces the two joystick potentiometers of the toy transmitter with SPI digital
 * pots, and "presses" its pushbuttons through MOSFETs/reed relays wired across the
 * switch contacts. The transmitter's RF link and the excavator are untouched.
 *
 * Protocol (ASCII lines, 115200 baud) -- keep in sync with control/transport.py:
 *   PING                       -> OK PONG
 *   KEEPALIVE                  -> OK
 *   NEUTRAL                    -> OK      all sticks to 128, all buttons released, all lines low
 *   SET_STICK <name> <0-255>   -> OK      128 = centred
 *   SET_LINE <name> <0|1>      -> OK      digital line low / high (W drive, O arm up, L cam tilt, A steer)
 *   PRESS_BUTTON <name> <ms>   -> OK      returns immediately; released after <ms>
 *   STATUS                     -> OK <stick=val ...> <button=0|1 ...>
 *   (errors)                   -> ERR <reason>
 *
 * SAFETY: if sticks are off-centre or a button is held and no command arrives for
 * WATCHDOG_MS, everything returns to neutral. The Python side sends KEEPALIVE while
 * it holds a motion, so a crashed script / unplugged USB cable stops the machine.
 *
 * Hardware notes (verify against your board before powering anything):
 *  - COMMON GROUND between the ESP32, the digipots and the transmitter board.
 *  - Power the digipots from the transmitter's own logic rail (or make sure the
 *    voltages agree); the wiper must never leave [GND, digipot VDD].
 *  - Each digipot's A/B terminals go to the same two rails the joystick pot spans;
 *    its wiper W goes where the joystick's wiper went. Lift/disconnect the physical
 *    pot's wiper (or use an analog switch) so the two aren't fighting.
 *  - MCP413x (MCP4131) has a shared SDI/SDO pin: wire it to MOSI; we never read.
 */

#include <SPI.h>

// ===========================================================================
//  USER CONFIG -- TODO: FILL IN after tracing the transmitter board.
//  -1 means "not assigned yet": the firmware will answer ERR unassigned_pin
//  for that control instead of driving some random pin.
// ===========================================================================

// Which digipot chip you are using.
#define DIGIPOT_MCP4131   1   // 7-bit, wiper 0..128, write = bytes {0x00, value}
#define DIGIPOT_MCP41010  2   // 8-bit, wiper 0..255, write = bytes {0x11, value}
#define DIGIPOT_TYPE      DIGIPOT_MCP4131   // TODO: set to the chip you actually have

// SPI bus pins. -1 = ESP32 default VSPI pins (SCK 18, MISO 19, MOSI 23).
const int SPI_SCK_PIN  = -1;
const int SPI_MISO_PIN = -1;
const int SPI_MOSI_PIN = -1;

// One digipot per joystick axis: chip-select GPIO for each.
// Names must match control/config.py (STICK_NAMES).
struct Stick { const char* name; int csPin; };
Stick STICKS[] = {
  {"left_x",  -1},   // TODO: CS GPIO
  {"left_y",  -1},   // TODO: CS GPIO
  {"right_x", -1},   // TODO: CS GPIO
  {"right_y", -1},   // TODO: CS GPIO
};

// One MOSFET / reed-relay driver per transmitter pushbutton you want to control.
// Rename to whatever the buttons on your transmitter actually do; the same names
// go in control/config.py (arm_axis etc.) and in `PRESS_BUTTON <name>` commands.
// activeHigh = true if a HIGH on the GPIO closes the switch (typical N-MOSFET).
struct Button { const char* name; int pin; bool activeHigh; };
Button BUTTONS[] = {
  {"boom_up",     -1, true},   // TODO: GPIO
  {"boom_down",   -1, true},   // TODO: GPIO
  {"arm_in",      -1, true},   // TODO: GPIO
  {"arm_out",     -1, true},   // TODO: GPIO
  {"bucket_curl", -1, true},   // TODO: GPIO
  {"bucket_dump", -1, true},   // TODO: GPIO
};

// The digital lines that drive the excavator directly, one per physical control (see the
// comment in control/config.py for what each does). activeHigh = true if a HIGH on the GPIO
// drives the line high. Names must match control/config.py (LINE_NAMES).
struct Line { const char* name; int pin; bool activeHigh; };
Line LINES[] = {
  {"W", -1, true},   // TODO: GPIO -- tracks forward
  {"O", -1, true},   // TODO: GPIO -- arm up
  {"L", -1, true},   // TODO: GPIO -- camera tilt
  {"A", -1, true},   // TODO: GPIO -- rotate tracks (steer)
};

const unsigned long WATCHDOG_MS  = 1000;   // must be > control/config.py keepalive_s
const unsigned long MAX_PRESS_MS = 5000;   // must be >= control/config.py max_press_ms
const unsigned long SPI_HZ       = 1000000;

// ===========================================================================
//  End of user config
// ===========================================================================

const uint8_t STICK_NEUTRAL = 128;
const int N_STICKS  = sizeof(STICKS)  / sizeof(STICKS[0]);
const int N_BUTTONS = sizeof(BUTTONS) / sizeof(BUTTONS[0]);
const int N_LINES   = sizeof(LINES)   / sizeof(LINES[0]);

#if DIGIPOT_TYPE == DIGIPOT_MCP4131
  const uint8_t DIGIPOT_MAX_STEP = 128;
#else
  const uint8_t DIGIPOT_MAX_STEP = 255;
#endif

uint8_t       stickValue[N_STICKS];
bool          buttonOn[N_BUTTONS];
bool          lineOn[N_LINES];
unsigned long buttonReleaseAt[N_BUTTONS];
unsigned long lastCommandMs = 0;

// ---- hardware ------------------------------------------------------------
void digipotWrite(int csPin, uint8_t value255) {
  uint8_t step = ((uint16_t)value255 * DIGIPOT_MAX_STEP + 127) / 255;
#if DIGIPOT_TYPE == DIGIPOT_MCP4131
  const uint8_t cmd = 0x00;   // address 0 (wiper 0), write
#else
  const uint8_t cmd = 0x11;   // write data, pot 0
#endif
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  digitalWrite(csPin, LOW);
  SPI.transfer(cmd);
  SPI.transfer(step);
  digitalWrite(csPin, HIGH);
  SPI.endTransaction();
}

void setStick(int i, uint8_t value) {
  stickValue[i] = value;
  if (STICKS[i].csPin >= 0) digipotWrite(STICKS[i].csPin, value);
}

void setButton(int i, bool on) {
  buttonOn[i] = on;
  if (BUTTONS[i].pin >= 0) digitalWrite(BUTTONS[i].pin, (on == BUTTONS[i].activeHigh) ? HIGH : LOW);
}

void setLine(int i, bool on) {
  lineOn[i] = on;
  if (LINES[i].pin >= 0) digitalWrite(LINES[i].pin, (on == LINES[i].activeHigh) ? HIGH : LOW);
}

void neutralAll() {
  for (int i = 0; i < N_STICKS; i++)  setStick(i, STICK_NEUTRAL);
  for (int i = 0; i < N_BUTTONS; i++) setButton(i, false);
  for (int i = 0; i < N_LINES; i++)   setLine(i, false);
}

bool anythingActive() {
  for (int i = 0; i < N_STICKS; i++)  if (stickValue[i] != STICK_NEUTRAL) return true;
  for (int i = 0; i < N_BUTTONS; i++) if (buttonOn[i]) return true;
  for (int i = 0; i < N_LINES; i++)   if (lineOn[i]) return true;
  return false;
}

int findStick(const char* n)  { for (int i = 0; i < N_STICKS; i++)  if (!strcmp(STICKS[i].name, n))  return i; return -1; }
int findButton(const char* n) { for (int i = 0; i < N_BUTTONS; i++) if (!strcmp(BUTTONS[i].name, n)) return i; return -1; }
int findLine(const char* n)   { for (int i = 0; i < N_LINES; i++)   if (!strcmp(LINES[i].name, n))   return i; return -1; }

// ---- command handling ----------------------------------------------------
bool parseUInt(const char* s, long maxVal, long& out) {
  if (!s || !*s) return false;
  char* end;
  long v = strtol(s, &end, 10);
  if (*end != '\0' || v < 0 || v > maxVal) return false;
  out = v;
  return true;
}

void handleLine(char* line) {
  char* cmd = strtok(line, " \t");
  if (!cmd) return;
  char* a = strtok(NULL, " \t");
  char* b = strtok(NULL, " \t");
  lastCommandMs = millis();

  if (!strcmp(cmd, "PING")) {
    Serial.println("OK PONG");
  } else if (!strcmp(cmd, "KEEPALIVE")) {
    Serial.println("OK");
  } else if (!strcmp(cmd, "NEUTRAL") || !strcmp(cmd, "STOP")) {
    neutralAll();
    Serial.println("OK");
  } else if (!strcmp(cmd, "SET_STICK")) {
    int i = a ? findStick(a) : -1;
    long v;
    if (i < 0)                            { Serial.println("ERR unknown_stick"); return; }
    if (!parseUInt(b, 255, v))            { Serial.println("ERR bad_value"); return; }
    if (STICKS[i].csPin < 0)              { Serial.println("ERR unassigned_pin"); return; }
    setStick(i, (uint8_t)v);
    Serial.println("OK");
  } else if (!strcmp(cmd, "SET_LINE")) {
    int i = a ? findLine(a) : -1;
    long v;
    if (i < 0)                            { Serial.println("ERR unknown_line"); return; }
    if (!parseUInt(b, 1, v))              { Serial.println("ERR bad_value"); return; }
    if (LINES[i].pin < 0)                 { Serial.println("ERR unassigned_pin"); return; }
    setLine(i, v == 1);
    Serial.println("OK");
  } else if (!strcmp(cmd, "PRESS_BUTTON")) {
    int i = a ? findButton(a) : -1;
    long ms;
    if (i < 0)                            { Serial.println("ERR unknown_button"); return; }
    if (!parseUInt(b, MAX_PRESS_MS, ms) || ms == 0) { Serial.println("ERR bad_duration"); return; }
    if (BUTTONS[i].pin < 0)               { Serial.println("ERR unassigned_pin"); return; }
    setButton(i, true);
    buttonReleaseAt[i] = millis() + (unsigned long)ms;
    Serial.println("OK");
  } else if (!strcmp(cmd, "STATUS")) {
    Serial.print("OK");
    for (int i = 0; i < N_STICKS; i++)  { Serial.print(' '); Serial.print(STICKS[i].name);  Serial.print('='); Serial.print(stickValue[i]); }
    for (int i = 0; i < N_BUTTONS; i++) { Serial.print(' '); Serial.print(BUTTONS[i].name); Serial.print('='); Serial.print(buttonOn[i] ? 1 : 0); }
    for (int i = 0; i < N_LINES; i++)   { Serial.print(' '); Serial.print(LINES[i].name);   Serial.print('='); Serial.print(lineOn[i] ? 1 : 0); }
    Serial.println();
  } else {
    Serial.println("ERR unknown_command");
  }
}

// ---- Arduino entry points ------------------------------------------------
void setup() {
  Serial.begin(115200);
  // Drive every control pin to its inactive level BEFORE anything else.
  for (int i = 0; i < N_BUTTONS; i++)
    if (BUTTONS[i].pin >= 0) { pinMode(BUTTONS[i].pin, OUTPUT); digitalWrite(BUTTONS[i].pin, BUTTONS[i].activeHigh ? LOW : HIGH); }
  for (int i = 0; i < N_LINES; i++)
    if (LINES[i].pin >= 0) { pinMode(LINES[i].pin, OUTPUT); digitalWrite(LINES[i].pin, LINES[i].activeHigh ? LOW : HIGH); }
  for (int i = 0; i < N_STICKS; i++)
    if (STICKS[i].csPin >= 0) { pinMode(STICKS[i].csPin, OUTPUT); digitalWrite(STICKS[i].csPin, HIGH); }
  SPI.begin(SPI_SCK_PIN, SPI_MISO_PIN, SPI_MOSI_PIN, -1);
  neutralAll();
  lastCommandMs = millis();
  Serial.println("READY excavator_control v1");
}

void loop() {
  static char buf[64];
  static uint8_t len = 0;
  static bool overflow = false;

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      buf[len] = '\0';
      if (overflow) Serial.println("ERR line_too_long");
      else if (len) handleLine(buf);
      len = 0; overflow = false;
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = c;
    } else {
      overflow = true;
    }
  }

  unsigned long now = millis();
  for (int i = 0; i < N_BUTTONS; i++)   // timed button releases (signed diff is millis()-wrap safe)
    if (buttonOn[i] && (long)(now - buttonReleaseAt[i]) >= 0) setButton(i, false);

  if (anythingActive() && (now - lastCommandMs) > WATCHDOG_MS) {
    neutralAll();
    Serial.println("WARN watchdog_neutral");
  }
}
