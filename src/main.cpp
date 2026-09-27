#include <Arduino.h>
#include <Servo.h>

const int PAN_SERVO_PIN = A0;
const int TILT_SERVO_PIN = A1;
const int SENSOR_PIN = A4;

// Same scan positions as the original project.
const float PAN_CENTER = 90;
const float TILT_CENTER = 90;
const float PAN_MIN = PAN_CENTER - 35;
const float PAN_MAX = PAN_CENTER + 30;
const float PAN_STEP = 1.5;
const float TILT_MIN = TILT_CENTER - 39;
const float TILT_MAX = TILT_CENTER;
const float TILT_STEP = 1.5;

const unsigned long SERVO_SETTLE_TIME = 150;
const unsigned long INITIAL_SETTLE_TIME = 1000;
const unsigned long RETURN_CENTER_SETTLE_TIME = 1000;
const float SERVO_MOVE_STEP = 5;
// Sharp GP2Y0A sensors update about every 38 ms; space readings so each one is new.
const int SENSOR_SAMPLES = 7;
const unsigned long SENSOR_SAMPLE_INTERVAL = 40;
// true: stop servo pulses while sampling so motor current cannot disturb the sensor.
// The noise test showed 0.17 V holding vs 0.015 V released, so this is on.
const bool RELEASE_WHILE_MEASURING = true;
const unsigned long RELEASE_SETTLE_TIME = 30;
const int NOISE_SAMPLES = 1000;
const unsigned long NOISE_INTERVAL = 10;
const float ADC_REFERENCE_V = 5.0;
const int SERVO_MIN_US = 544;
const int SERVO_MAX_US = 2400;

Servo panServo, tiltServo;
float currentPan = PAN_CENTER;
float currentTilt = TILT_CENTER;

void writeAngle(Servo &servo, float angle) {
    // Servo::write takes an integer. Pulse width preserves fractional angles.
    servo.writeMicroseconds(lround(SERVO_MIN_US + angle * (SERVO_MAX_US - SERVO_MIN_US) / 180.0));
}

void attachServos() {
    writeAngle(panServo, currentPan);
    writeAngle(tiltServo, currentTilt);
    panServo.attach(PAN_SERVO_PIN);
    tiltServo.attach(TILT_SERVO_PIN);
}

void stopMotors() {
    panServo.detach();
    tiltServo.detach();
    Serial.println(F("MOTORS_OFF"));
}

bool waitWithStop(unsigned long duration) {
    const unsigned long start = millis();
    do {
        while (Serial.available()) {
            const char command = Serial.read();
            if (command == 'x' || command == 'X') return false;
        }
        delay(1);
    } while (millis() - start < duration);
    return true;
}

float stepToward(float value, float target) {
    if (fabs(target - value) <= SERVO_MOVE_STEP) return target;
    return value + (target > value ? SERVO_MOVE_STEP : -SERVO_MOVE_STEP);
}

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

float voltage(int reading) {
    return reading * ADC_REFERENCE_V / 1023.0;
}

bool measure(float pan, float tilt) {
    if (RELEASE_WHILE_MEASURING) {
        panServo.detach();
        tiltServo.detach();
        if (!waitWithStop(RELEASE_SETTLE_TIME)) return false;
    }
    // Median rejects single spikes that a mean would smear into the result.
    int readings[SENSOR_SAMPLES];
    for (int i = 0; i < SENSOR_SAMPLES; ++i) {
        const int value = analogRead(SENSOR_PIN);
        int j = i;
        for (; j > 0 && readings[j - 1] > value; --j) readings[j] = readings[j - 1];
        readings[j] = value;
        if (i < SENSOR_SAMPLES - 1 && !waitWithStop(SENSOR_SAMPLE_INTERVAL)) return false;
    }
    if (RELEASE_WHILE_MEASURING) attachServos();
    Serial.print(F("DATA,"));
    Serial.print(pan, 1);
    Serial.print(',');
    Serial.print(tilt, 1);
    Serial.print(',');
    Serial.println(voltage(readings[SENSOR_SAMPLES / 2]), 4);
    return true;
}

// Stream raw readings at the center: phase 1 = servos holding, phase 0 = released.
bool streamRaw(int phase) {
    const unsigned long start = millis();
    for (int i = 0; i < NOISE_SAMPLES; ++i) {
        const int value = analogRead(SENSOR_PIN);
        Serial.print(F("DATA,"));
        Serial.print(phase);
        Serial.print(',');
        Serial.print(millis() - start);
        Serial.print(',');
        Serial.println(voltage(value), 4);
        if (!waitWithStop(NOISE_INTERVAL)) return false;
    }
    return true;
}

bool noiseTest() {
    if (!moveTo(PAN_CENTER, TILT_CENTER) || !waitWithStop(INITIAL_SETTLE_TIME)) return false;
    if (!streamRaw(1)) return false;
    panServo.detach();
    tiltServo.detach();
    return waitWithStop(500) && streamRaw(0);
}

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
    } else if (command == 'x' || command == 'X') {
        stopMotors();
        Serial.println(F("SCAN_ABORTED"));
    }
}
