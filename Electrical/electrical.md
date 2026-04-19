# Electrical Documentation

## Components Used

### TurtleBot Components
- OpenCR (32-bit ARM Cortex-M7)
- Raspberry Pi 4B
- 2x DYNAMIXEL XL430-W250 Motors
- 11.1V LiPo Battery
- L298N Motor Driver
- SG90 Servo Motor
- RS360 DC Motor
- HC-SR04 Ultrasonic Sensor
- 720p USB Camera
- LDS-2 360° LiDAR

## Connections

### SG90 Servo
- VCC → 5V rail from Raspberry Pi
- GND → Common ground
- Signal → GPIO12 (hardware PWM) on Raspberry Pi

### L298N Motor Driver

#### Power
- VCC → 12V rail from OpenCR
- GND → Common ground (OpenCR)

#### Control Signals
- ENA (Enable Pin) → GPIO13 (hardware PWM) on Raspberry Pi (for speed control)
- IN1 → GPIO25 on Raspberry Pi (direction control)
- IN2 → GPIO8 on Raspberry Pi (direction control)

#### Motor Output
- OUT1, OUT2 → Connected to the two terminals of the RS360 DC motor

### HC-SR04 Ultrasonic Sensor
- VCC → 5V rail from Raspberry Pi
- GND → Common ground
- TRIG → GPIO24 on Raspberry Pi
- ECHO → GPIO23 on Raspberry Pi via voltage divider (1kΩ and 2kΩ resistors, 5V → 3.3V)

### DYNAMIXEL XL430-W250 Motors
- Connected to OpenCR via TTL serial (3-pin JST connector)
- OpenCR manages all Dynamixel communication and control internally

### USB Camera
- Connected directly to a USB port on the Raspberry Pi

### LDS-2 LiDAR
- LiDAR ↔ USB2LDS via UART
- USB2LDS → Raspberry Pi via USB

### OpenCR
- Connected to Raspberry Pi via USB cable
- Appears as /dev/ttyACM0 on the Raspberry Pi
- Runs the TurtleBot3 ROS 2 serial bridge, exposing wheel odometry and motor control as ROS 2 topics

## Wiring Diagram

![Electrical Wiring Diagram](Electrical%20Diagrams/Wiring%20Diagram.png)

## Raspberry Pi Pin Mapping

Pin numbering follows the BCM (Broadcom SOC channel) convention.
The OpenCR board connects to the Raspberry Pi via USB (`/dev/ttyACM0`) — it does not use any GPIO pin.

| Category | Function           | GPIO (BCM) | Physical Pin | Description                          |
|----------|--------------------|------------|--------------|--------------------------------------|
| PWM      | Servo Signal       | GPIO12     | Pin 32       | Hardware PWM control for SG90 servo  |
| PWM      | Motor Enable (ENA) | GPIO13     | Pin 33       | Hardware PWM speed control for L298N |
| Motor    | Direction IN1      | GPIO25     | Pin 22       | Direction control input              |
| Motor    | Direction IN2      | GPIO8      | Pin 24       | Direction control input              |
| Sensor   | Ultrasonic TRIG    | GPIO24     | Pin 18       | Trigger signal                       |
| Sensor   | Ultrasonic ECHO    | GPIO23     | Pin 16       | Via voltage divider (5V → 3.3V)      |

## Power and Communication Architecture

This section outlines how power is distributed throughout the system and how signals are communicated between components.

### Power Architecture

![Power Architecture](Electrical%20Diagrams/Power%20Architecture.png)
*Figure: Power distribution across system components*

- The system is powered by an 11.1V LiPo battery connected to the OpenCR board.
- The OpenCR board distributes power to high-power components such as the L298N motor driver, Raspberry Pi, and DYNAMIXEL motors.
- The Raspberry Pi provides a regulated 5V supply to low-power components including the SG90 servo, HC-SR04 ultrasonic sensor, and USB peripherals.
- High-power components (e.g., RS360 motor) draw power through the L298N motor driver, which is supplied directly from the OpenCR battery rail.
- All components share a common ground to ensure a consistent voltage reference and reliable operation.

---

### Communication Architecture

![Communication Architecture](Electrical%20Diagrams/Communication%20Architecture.png)
*Figure: Communication and signal flow between system components*

- The Raspberry Pi 4B acts as the main controller, handling sensor processing, decision-making, and overall system logic.
- Communication interfaces are structured as follows:
  - **UART (bidirectional):**
    - LDS-2 LiDAR ↔ USB2LDS — raw UART sensor data, converted to USB for the Raspberry Pi
  - **TTL Serial:**
    - OpenCR ↔ DYNAMIXEL XL430-W250 motors — OpenCR manages all Dynamixel communication internally via the TTL bus
  - **GPIO / PWM (low-level control signals):**
    - L298N motor driver (direction via GPIO25/GPIO8, speed via GPIO13 PWM)
    - SG90 servo (GPIO12 hardware PWM)
    - HC-SR04 ultrasonic sensor (TRIG/ECHO via GPIO24/GPIO23)
  - **USB (high-bandwidth data):**
    - OpenCR → Raspberry Pi via USB (/dev/ttyACM0). Runs the TurtleBot3 ROS 2 serial bridge over USB-serial.
    - USB camera → Raspberry Pi (image data)
    - USB2LDS → Raspberry Pi (LiDAR data)

## Power Budget Analysis

A power budget analysis was conducted to ensure that the system can operate reliably within the limits of the onboard battery.

### System Assumptions
- Power Source: 11.1V 3S LiPo Battery (1800mAh, ~19.98 Wh)
- Mission Duration: 25 minutes per run

### Key Results
- Total Energy Consumption per Mission: ~4.98 Wh
- Estimated Number of Missions per Charge: ~4 full cycles

### Detailed Analysis
[View Full Power Budget Analysis](Power%20Budget%20Analysis.pdf)

## Notes
- Ensure all components share a **common ground** (OpenCR, Raspberry Pi, sensors, and L298N)
- The voltage divider on the ECHO pin uses a **1kΩ and 2kΩ resistor** in series to step 5V down to ~3.3V — connecting the ECHO pin directly without this will damage the Raspberry Pi GPIO
- The OpenCR ↔ Raspberry Pi UART link runs the TurtleBot3 bringup (`ros2 launch turtlebot3_bringup robot.launch.py`) — this must be running for `/cmd_vel` and `/odom` topics to be active
