import RPi.GPIO as GPIO
import time

# ── Pin Config ──
SERVO_PIN   = 12
ENABLE_PIN  = 13
INT1        = 25
INT2        = 8

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN,  GPIO.OUT)
GPIO.setup(ENABLE_PIN, GPIO.OUT)
GPIO.setup(INT1,       GPIO.OUT)
GPIO.setup(INT2,       GPIO.OUT)

# SG90: 50Hz, duty cycle 2.5% = 0°, 7.5% = 90°, 12.5% = 180°
servo = GPIO.PWM(SERVO_PIN, 50)
motor = GPIO.PWM(ENABLE_PIN, 1000)

servo.start(0)
motor.start(0)

def set_servo(angle):
    duty = 2.5 + (angle / 180.0) * 10.0
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)  # stop jitter

def shoot():
    print("Opening gate — letting ball in...")
    set_servo(90)
    time.sleep(0.5)

    print("Closing gate...")
    set_servo(0)
    time.sleep(0.3)

    print("Firing motor for 2s...")
    GPIO.output(INT1, GPIO.HIGH)
    GPIO.output(INT2, GPIO.LOW)
    motor.ChangeDutyCycle(100)  # full speed
    time.sleep(2.0)

    print("Stopping motor.")
    motor.ChangeDutyCycle(0)
    GPIO.output(INT1, GPIO.LOW)

try:
    shoot()
finally:
    servo.stop()
    motor.stop()
    GPIO.cleanup()
    print("Done.")

