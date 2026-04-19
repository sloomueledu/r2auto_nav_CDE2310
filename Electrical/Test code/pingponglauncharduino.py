#include <Servo.h>

// ── Pin Config ──
#define SERVO_PIN   9
#define ENABLE_PIN  10   // PWM pin for L298N enable
#define INT1        7
#define INT2        8

Servo servo;

void setServo(int angle) {
  servo.write(angle);
  delay(500);
}

void shoot() {
  Serial.println("Opening gate — letting ball in...");
  setServo(90);
  delay(500);

  Serial.println("Closing gate...");
  setServo(0);
  delay(300);

  Serial.println("Firing motor for 2s...");
  digitalWrite(INT1, HIGH);
  digitalWrite(INT2, LOW);
  analogWrite(ENABLE_PIN, 255);  // full speed
  delay(2000);

  Serial.println("Stopping motor.");
  analogWrite(ENABLE_PIN, 0);
  digitalWrite(INT1, LOW);
  digitalWrite(INT2, LOW);
}

void setup() {
  Serial.begin(9600);
  servo.attach(SERVO_PIN);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(INT1, OUTPUT);
  pinMode(INT2, OUTPUT);

  servo.write(0);  // start closed
  delay(500);

  shoot();
}

void loop() {
  // press 's' in Serial Monitor to shoot again
  if (Serial.available() && Serial.read() == 's') {
    shoot();
  }
}

