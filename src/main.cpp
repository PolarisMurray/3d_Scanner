// Pan/tilt IR scanner firmware (Arduino Uno).
//
// Serial protocol, 115200 baud:
//   host -> board   's' scan, 'n' noise test, 'x' stop
//   board -> host   READY, SCAN_START, DATA,..., SCAN_COMPLETE | SCAN_ABORTED, MOTORS_OFF
//   scan rows:  DATA,<pan_deg>,<tilt_deg>,<volts>
//   noise rows: DATA,<phase>,<ms>,<volts>   (phase 1 = servos holding, 0 = released)
// The host (python/acquisition.py) converts volts to distance and does all geometry.

#include <Arduino.h>
#include <Servo.h>

const int PAN_SERVO_PIN = A0;
const int TILT_SERVO_PIN = A1;
const int SENSOR_PIN = A4;

// Scan grid in servo degrees. python/settings.py PAN_STEP_DEG / TILT_STEP_DEG must match.
const float PAN_CENTER = 90;
const float TILT_CENTER = 90;
const float PAN_MIN = PAN_CENTER - 35;
const float PAN_MAX = PAN_CENTER + 30;
const float PAN_STEP = 1.5;
const float TILT_MIN = TILT_CENTER - 39;
const float TILT_MAX = TILT_CENTER + 5;
const float TILT_STEP = 1.5;

// Servo motion: move at most SERVO_MOVE_STEP degrees per SERVO_SETTLE_TIME ms.
const unsigned long SERVO_SETTLE_TIME = 150;
const unsigned long INITIAL_SETTLE_TIME = 1000;
const unsigned long RETURN_CENTER_SETTLE_TIME = 1000;
const float SERVO_MOVE_STEP = 5;
const int SERVO_MIN_US = 544;
const int SERVO_MAX_US = 2400;

// Sharp GP2Y0A sensors update about every 38 ms; space readings so each one is new.
// python/noise.py NEW_SAMPLES / NEW_SPACING_MS mirror these two values.
const int SENSOR_SAMPLES = 7;
const unsigned long SENSOR_SAMPLE_INTERVAL = 40;
// true: stop servo pulses while sampling so motor current cannot disturb the sensor.
// The noise test showed 0.17 V holding vs 0.015 V released, so this is on.
const bool RELEASE_WHILE_MEASURING = true;
const unsigned long RELEASE_SETTLE_TIME = 30;
const float ADC_REFERENCE_V = 5.0;

// Noise test: NOISE_SAMPLES raw readings per phase, NOISE_INTERVAL ms apart.
const int NOISE_SAMPLES = 1000;
const unsigned long NOISE_INTERVAL = 10;
const unsigned long NOISE_PHASE_GAP = 500;
const int PHASE_HOLDING = 1;
const int PHASE_RELEASED = 0;

Servo panServo, tiltServo;
float currentPan = PAN_CENTER;
float currentTilt = TILT_CENTER;

bool isStopCommand(char command) {
    return command == 'x' || command == 'X';
}

void writeAngle(Servo &servo, float angle) {
    // Servo::write takes an integer. Pulse width preserves fractional angles.
    servo.writeMicroseconds(lround(SERVO_MIN_US + angle * (SERVO_MAX_US - SERVO_MIN_US) / 180.0));
}

void attachServos() {
    // Set the pulse first so attaching does not jump to a default angle.
    writeAngle(panServo, currentPan);
    writeAngle(tiltServo, currentTilt);
    panServo.attach(PAN_SERVO_PIN);
    tiltServo.attach(TILT_SERVO_PIN);
}

void detachServos() {
    panServo.detach();
    tiltServo.detach();
}

void stopMotors() {
    detachServos();
    Serial.println(F("MOTORS_OFF"));
}

// Wait `duration` ms while watching for a stop command. Returns false if stopped.
bool waitWithStop(unsigned long duration) {
    const unsigned long start = millis();
    do {
        while (Serial.available()) {
            if (isStopCommand(Serial.read())) return false;
        }
        delay(1);
    } while (millis() - start < duration);
    return true;
}

float stepToward(float value, float target) {
    if (fabs(target - value) <= SERVO_MOVE_STEP) return target;
    return value + (target > value ? SERVO_MOVE_STEP : -SERVO_MOVE_STEP);
}

// Move both servos in small steps to limit current spikes and overshoot.
bool moveTo(float pan, float tilt) {
    while (currentPan != pan || currentTilt != tilt) {
        currentPan = stepToward(currentPan, pan);
        currentTilt = stepToward(currentTilt, tilt);
        writeAngle(panServo, currentPan);
        writeAngle(tiltServo, currentTilt);
        if (!waitWithStop(SERVO_SETTLE_TIME)) return false;
    }
    return true;
}

float toVolts(int reading) {
    return reading * ADC_REFERENCE_V / 1023.0;
}

// Median of SENSOR_SAMPLES readings: rejects single spikes that a mean would smear in.
bool readMedian(int &median) {
    int sorted[SENSOR_SAMPLES];
    for (int i = 0; i < SENSOR_SAMPLES; ++i) {
        // Insertion sort as the readings arrive.
        const int value = analogRead(SENSOR_PIN);
        int j = i;
        for (; j > 0 && sorted[j - 1] > value; --j) sorted[j] = sorted[j - 1];
        sorted[j] = value;
        if (i < SENSOR_SAMPLES - 1 && !waitWithStop(SENSOR_SAMPLE_INTERVAL)) return false;
    }
    median = sorted[SENSOR_SAMPLES / 2];
    return true;
}

bool measure(float pan, float tilt) {
    if (RELEASE_WHILE_MEASURING) {
        detachServos();
        if (!waitWithStop(RELEASE_SETTLE_TIME)) return false;
    }
    int median;
    if (!readMedian(median)) return false;
    if (RELEASE_WHILE_MEASURING) attachServos();
    Serial.print(F("DATA,"));
    Serial.print(pan, 1);
    Serial.print(',');
    Serial.print(tilt, 1);
    Serial.print(',');
    Serial.println(toVolts(median), 4);
    return true;
}

bool streamRaw(int phase) {
    const unsigned long start = millis();
    for (int i = 0; i < NOISE_SAMPLES; ++i) {
        const int value = analogRead(SENSOR_PIN);
        Serial.print(F("DATA,"));
        Serial.print(phase);
        Serial.print(',');
        Serial.print(millis() - start);
        Serial.print(',');
        Serial.println(toVolts(value), 4);
        if (!waitWithStop(NOISE_INTERVAL)) return false;
    }
    return true;
}

// Raw readings at the center, first with servos powered, then released.
bool noiseTest() {
    if (!moveTo(PAN_CENTER, TILT_CENTER) || !waitWithStop(INITIAL_SETTLE_TIME)) return false;
    if (!streamRaw(PHASE_HOLDING)) return false;
    detachServos();
    return waitWithStop(NOISE_PHASE_GAP) && streamRaw(PHASE_RELEASED);
}

// Serpentine raster: tilt runs up one pan column and down the next, so no long return moves.
bool scan() {
    const int columns = ceil((PAN_MAX - PAN_MIN) / PAN_STEP) + 1;
    const int rows = ceil((TILT_MAX - TILT_MIN) / TILT_STEP) + 1;
    if (!moveTo(PAN_MIN, TILT_MIN) || !waitWithStop(INITIAL_SETTLE_TIME)) return false;
    for (int col = 0; col < columns; ++col) {
        const float pan = min(PAN_MIN + col * PAN_STEP, PAN_MAX);
        for (int row = 0; row < rows; ++row) {
            const int index = col % 2 == 0 ? row : rows - 1 - row;
            const float tilt = min(TILT_MIN + index * TILT_STEP, TILT_MAX);
            if (!moveTo(pan, tilt) || !measure(pan, tilt)) return false;
        }
    }
    return moveTo(PAN_CENTER, TILT_CENTER) && waitWithStop(RETURN_CENTER_SETTLE_TIME);
}

void setup() {
    Serial.begin(115200);
    Serial.println(F("READY"));
}

void loop() {
    if (!Serial.available()) return;
    const char command = Serial.read();
    if (command == 's' || command == 'n') {
        attachServos();
        Serial.println(F("SCAN_START"));
        const bool complete = command == 's' ? scan() : noiseTest();
        stopMotors();
        Serial.println(complete ? F("SCAN_COMPLETE") : F("SCAN_ABORTED"));
    } else if (isStopCommand(command)) {
        stopMotors();
        Serial.println(F("SCAN_ABORTED"));
    }
}
