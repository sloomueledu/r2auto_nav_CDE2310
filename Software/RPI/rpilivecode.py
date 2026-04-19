import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import RPi.GPIO as GPIO
import time

# PWM CONFIGURATION
SERVO_PIN = 12

# L298N MOTOR DRIVER ENABLE PINS
ENABLE_PIN = 13

# L298N MOTOR DRIVER PINS
INT1 = 25 
INT2 = 8 
 
# ULTRASONIC SENSOR PINS 
TRIG = 24 
ECHO = 23 
 
class RPILiveCode(Node): 
    def __init__(self): 
        super().__init__('rpi_live_code') 
 
        # GPIO SETUP 
        GPIO.setmode(GPIO.BCM) 
        GPIO.setup(SERVO_PIN, GPIO.OUT) 
        GPIO.setup(ENABLE_PIN, GPIO.OUT) 
        GPIO.setup(INT1, GPIO.OUT) 
        GPIO.setup(INT2, GPIO.OUT) 
        GPIO.setup(TRIG, GPIO.OUT) 
        GPIO.setup(ECHO, GPIO.IN) 
 
        # SERVO & ENABLE PINS PWM SETUP 
        self.servo_pwm = GPIO.PWM(SERVO_PIN, 50.00)  # 50Hz for servo 
        self.enable_pwm = GPIO.PWM(ENABLE_PIN, 100)  # 100Hz for motor speed control 
        self.enable_pwm.start(0)  # Start with motors stopped 
        self.servo_pwm.start(2.5)  # Reset servo to starting position 
        time.sleep(0.5) 
        self.servo_pwm.ChangeDutyCycle(0)  # Stop sending signal to servo 
 
        # ROS2 SUBSCRIBER 
        self.subscription = self.create_subscription( 
            String, 
            'gpio_commands', 
            self.launch_command, 
            10 
        ) 
        self.subscription  # prevent unused variable warning 
 
        # EXECUTED PUBLISHER 
        self.piResponse = self.create_publisher(String, 'rpi_response', 10) 
 
        # VARIABLES 
        self.dropBallangle = 90 
        self.reset = 0 
        self.move_duration = 0.5 
        self.reset_duration = 0.5 
        self.UltSenWait = 0.5 
        self.DetStaWait = 0.3 
        self.speed = 30 # Speed percentage (0-100) 
        self.move_flywheel_duration = 1.25  # Duration to run the flywheel in seconds 
        self.command = None 
        self.StationATracker = 0 
        self.StationBTracker = 0 
        self.StationAFLAG = False 
        self.StationBFLAG = False 
        self.path_clear = False  # For Station B firing logic 
 
        # TIMERS 
        self.servo_timer = None 
        self.flywheel_timer = None 
        self.StationAtimer = None 
        self.reset_timer = None  
        self.StationB_tripwire_timer = None 
 
        # STATE CONTROL 
        self.state = 'idle'  # Possible states: 'idle', 'launching', 'waiting', 'completed', 'distance_check' 
 
        # L298N IN PIN CONFIGURATION 
        GPIO.output(INT1, GPIO.LOW) 
        GPIO.output(INT2, GPIO.HIGH) 
 
    def distance(self): 
        GPIO.output(TRIG, False) 
        time.sleep(0.000002)  # Fast settle 
 
        GPIO.output(TRIG, True) 
        time.sleep(0.00001) 
        GPIO.output(TRIG, False) 
 
        pulse_start = time.time() 
        pulse_end = time.time() 
         
        max_time = time.time() + 0.04 # 40ms timeout safety net 
 
        while GPIO.input(ECHO) == 0: 
            pulse_start = time.time()
            if pulse_start > max_time:
                return 814.00 # Return a safe "far away" distance if it glitches

        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
            if pulse_end > max_time:
                return 814.00

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        return round(distance, 2)
    """
    def determine_station(self):
        
        if self.StationAFLAG and not self.StationBFLAG:
            self.get_logger().info('Station A already completed. Skipping to Station B.')
            return 'B'

        elif self.StationBFLAG and not self.StationAFLAG:
            self.get_logger().info('Station B already completed. Skipping to Station A.')
            return 'A'
              
        else:
            self.get_logger().info('Analyzing station environment...')
            
            # Take 3 distance measurements spaced out over time
            dist1 = self.distance()
            time.sleep(self.DetStaWait)
            dist2 = self.distance()
            time.sleep(self.DetStaWait)
            dist3 = self.distance()

            self.get_logger().info(f'Scan results: {dist1}cm, {dist2}cm, {dist3}cm')

            # 1. STATION A CHECK: Is it Close AND Stable?
            # If all 3 readings are under 30cm and haven't moved more than 2cm...
            if dist1 < 30.0 and abs(dist1 - dist2) < 2.0 and abs(dist2 - dist3) < 2.0:
                self.get_logger().info('Stationary target detected directly ahead. Confirming Station A.')
                return 'A'

            # 2. STATION B CHECK: Is the track empty OR is the object moving?
            # If any reading sees the far wall (>30cm) OR the distance changed drastically (movement)...
            elif dist1 > 30.0 or dist2 > 30.0 or dist3 > 30.0 or abs(dist1 - dist3) >= 2.0:
                self.get_logger().info('Track is clear or target is moving. Confirming Station B.')
                
                # Reset the tripwire safety just in case the cart was passing during the scan
                self.path_clear = False 
                return 'B'

            # 3. SAFETY NET: If the sensor glitches out
            else:
                self.get_logger().info('Sensor readings inconclusive. Aborting launch sequence.')
                return None
    """
    def launch_command(self, msg):
        self.command = msg.data
        self.get_logger().info(f'Received command: {self.command}')
        if self.state == 'idle':
            self.launchControl()

    def launchControl(self):
        if self.command == 'A' and self.state == 'idle' and not self.StationAFLAG:
            self.state = 'launching'
            self.ExecuteStationA()

        elif self.command == 'B' and self.state == 'idle' and not self.StationBFLAG:
            self.state = 'launching'
            self.ExecuteStationB()

        elif self.state != 'idle':
            self.get_logger().info('Command ignored: Robot is currently busy launching.')

        elif self.command is not None:
            self.get_logger().info(f'Command ignored: Station {self.command} is already completed.')
        else:
            self.get_logger().info('Command ignored: Invalid Input/Error Occurred.')

    def ExecuteStationA(self):
        self.get_logger().info('Executing Station A sequence %d' % (self.StationATracker))

        # CANCEL STRAY TIMERS
        if self.StationAtimer is not None:
            self.StationAtimer.cancel()
            self.StationAtimer = None

        if self.StationATracker == 0:
            self.StationATracker += 1
            self.move_flywheel(self.speed)
            time.sleep(0.5)
            self.trigger_drop()

            # PREP FOR SECOND DROP
            self.get_logger().info('1st Payload Out. Waiting for 2nd payload...')
            self.StationAtimer = self.create_timer(7.5, self.ExecuteStationA)

        elif self.StationATracker == 1:
            self.StationATracker += 1
            self.move_flywheel_duration = 4
            self.move_flywheel(self.speed)
            time.sleep(0.5)
            self.trigger_drop()

            # PREP FOR FINAL DROP
            self.get_logger().info('2nd Payload Out. Waiting for final payload...')
            self.StationAtimer = self.create_timer(2.0, self.ExecuteStationA)

        elif self.StationATracker == 2:
            self.StationATracker = 0
            self.StationAFLAG = True
            self.move_flywheel_duration = 0.75
            self.trigger_drop()

            # RESET STATE
            self.state = 'idle'
            self.command = None

            # HAND CONTROL BACK TO LAPTOP
            self.get_logger().info('Station A sequence complete.')
            self.piResponse.publish(String(data="A COMPLETE"))

    def determineifFireB(self):
        distance = self.distance()
        self.get_logger().info(f'Measured distance for Station B: {distance} cm')
        if distance >= 30.00:
            self.get_logger().info('No Receptacle Detected. Waiting,,,')
            self.path_clear = True
            return False
        elif distance < 15.00 and self.path_clear:
            self.get_logger().info('Leading Edge of Receptacle Found! Firing... [Diagnostic: Distance = %.2f cm]' % distance)
            self.path_clear = False
            return True
        else:
            self.get_logger().info('Waiting for Receptacle to Appear/Clear... [Diagnostic: Distance = %.2f cm]' % distance)
            return False


    def ExecuteStationB(self):
        self.get_logger().info('Station B sequence started. Arming tripwire...')
        self.move_flywheel_duration = 120
        self.move_flywheel(self.speed)
        # Start a fast-repeating timer (every 0.05s) to check the distance WITHOUT blocking ROS2
        self.StationB_tripwire_timer = self.create_timer(0.1, self.tripwire_loop)

    def tripwire_loop(self):
        if self.StationBTracker < 3:
            if self.determineifFireB():
                self.trigger_drop()
                self.StationBTracker += 1
                self.get_logger().info(f'Payload {self.StationBTracker} launched. Waiting for next payload...')
            else:
                pass
        else:
            self.stop_flywheel()  # Ensure flywheel is stopped after 3 payloads
            # We hit 3 payloads! Clean up the timer and finish the sequence.
            self.StationB_tripwire_timer.cancel()
            self.StationB_tripwire_timer = None

            self.state = 'idle'
            self.StationBTracker = 0
            self.StationBFLAG = True

            self.get_logger().info('Station B sequence complete.')
            self.piResponse.publish(String(data="B COMPLETE"))

    def trigger_drop(self):
        """Initiates the drop sequence."""
        self.get_logger().info('Dropping payload...')

        # CANCEL ANY OLD TIMERS
        if self.servo_timer is not None:
            self.servo_timer.cancel()
            self.servo_timer = None

        # 1. MOVE TO DROP ANGLE (0 degrees)
        duty_cycle = 2.5 + (self.dropBallangle / 18.0)
        self.servo_pwm.ChangeDutyCycle(duty_cycle)

        # 2. START A 0.5s TIMER TO IMMEDIATELY RETRACT
        if self.reset_timer is not None:
            self.reset_timer.cancel()
        self.reset_timer = self.create_timer(0.5, self.retract_servo)

    def retract_servo(self):
        """Immediately pulls the servo back to 90 degrees."""
        self.get_logger().info('Resetting servo to starting position.')

        # 3. MOVE BACK TO 90 DEGREES
        duty_cycle = 2.5 + (self.reset / 18.0)
        self.servo_pwm.ChangeDutyCycle(duty_cycle)

        # Stop the reset timer so it only fires once
        if self.reset_timer is not None:
            self.reset_timer.cancel()
            self.reset_timer = None

        # 4. START A 0.5s TIMER TO CUT POWER TO THE SERVO
        self.servo_timer = self.create_timer(0.5, self.stop_servo)

    def stop_servo(self):
        """Cuts PWM power to the servo to prevent jittering."""
        self.servo_pwm.ChangeDutyCycle(0)

        if self.servo_timer is not None:
            self.servo_timer.cancel()
            self.servo_timer = None

    def move_flywheel(self, speed_percentage):
        if self.flywheel_timer is not None:
            self.flywheel_timer.cancel()
            self.flywheel_timer = None

        duty_cycle = max(0, min(100, (speed_percentage)))
        self.enable_pwm.ChangeDutyCycle(duty_cycle)

        if speed_percentage > 0:
            self.flywheel_timer = self.create_timer(self.move_flywheel_duration, self.stop_flywheel)

    def stop_flywheel(self):
        self.enable_pwm.ChangeDutyCycle(0)
        if self.flywheel_timer is not None:
            self.flywheel_timer.cancel()
            self.flywheel_timer = None

    def resetAll(self, msg=None):
        self.get_logger().info('Resetting all components to default state')
        self.servo_pwm.ChangeDutyCycle(0)
        self.enable_pwm.ChangeDutyCycle(0)
        time.sleep(0.5)
        self.state = 'idle'
        self.command = None
        self.StationATracker = 0
        self.StationBTracker = 0

        self.StationAFLAG = False
        self.StationBFLAG = False
        self.path_clear = False

        if self.reset_timer is not None:
            self.reset_timer.cancel()
            self.reset_timer = None

    def clear_gpio(self):
        GPIO.cleanup()

def main(args=None):
    gpio_controller = None
    try:
        rclpy.init(args=args)
        gpio_controller = RPILiveCode()
        rclpy.spin(gpio_controller)
    except (KeyboardInterrupt , Exception) as e:
        print(f"Exception occurred: {e}")
        print("Shutting down gracefully...")
        if gpio_controller is not None:
            gpio_controller.resetAll()
    finally:
        if gpio_controller is not None:
            gpio_controller.clear_gpio()
            gpio_controller.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()


