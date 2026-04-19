# Electrical Documentation

## Components Used
### TurtleBot Components
- OpenCR (32-bit ARM Cortex-M7)
- Raspberry Pi
- DYNAMIXEL Motor
- 11.1V LiPo Battery
- L298N Motor Driver
- SG90 Servo Motor
- RS360 DC Motor
- HC-SR04 Ultrasonic Sensor
- 720p USB Camera
- 360 degree LiDAR

## Connections
### SG90 Servo
- VCC → 5V rail from Raspberry Pi  
- GND → Common ground  
- Signal → PWM-capable GPIO pin on Raspberry Pi  

### L298N Motor Driver
#### Power
- VCC → 12V rail from OpenCR  
- GND → Common ground (OpenCR)
#### Control Signals
- ENA (Enable Pin) → PWM-capable GPIO pin on Raspberry Pi (for speed control)  
- IN1, IN2 → Digital GPIO pins on Raspberry Pi (for direction control)  
#### Motor Output
- OUT1, OUT2 → Connected to the two terminals of the RS360 DC motor  

### HC-SR04 Ultrasonic Sensor
- VCC → 5V rail from Raspberry Pi  
- GND → Common ground  
- TRIG → Digital GPIO pin on Raspberry Pi  
- ECHO → Connected to GPIO via voltage divider (5V → 3.3V)

### USB Camera
- Connected directly to the USB port of the Raspberry Pi

### LiDAR
- Connected through USB2LDS to the USB port of the Raspberry Pi  

## Wiring Diagram
![Electrical Wiring Diagram](Electrical%20Diagrams/Wiring%20Diagram.png)

## Raspberry Pi Pin Mapping
Pin numbering follows the BCM (Broadcom SOC channel) convention
| Category     | Function              | GPIO (BCM) | Description                       |
|-------------|----------------------|-----------|-----------------------------------|
| PWM         | Servo Signal         | GPIO12    | PWM control for SG90 servo        |
| PWM         | Motor Enable (ENA)   | GPIO13    | PWM speed control for L298N       |
| Motor       | Direction IN1        | GPIO25    | Direction control input           |
| Motor       | Direction IN2        | GPIO8     | Direction control input           |
| Sensor      | Ultrasonic TRIG      | GPIO24    | Trigger signal                    |
| Sensor      | Ultrasonic ECHO      | GPIO23    | Via voltage divider (5V → 3.3V)   |

## Power and Communication Architecture
This section outlines how power is distributed throughout the system and how signals are communicated between components.

### Power Architecture
![Power Architecture](Electrical%20Diagrams/Power%20Architecture.png)

*Figure: Power distribution across system components*

- The system is powered by an 11.1V LiPo battery connected to the OpenCR board.
- The OpenCR board distributes power to high-power components such as the L298N motor driver, RaspberryPi, and Dynamixel Motors.
- The Raspberry Pi provides a regulated 5V supply to low-power components including the SG90 servo, HC-SR04 ultrasonic sensor, and USB peripherals.
- High-power components (e.g., RS360 motor) draw power through the L298N motor driver, which is supplied by the OpenCR.
- All components share a common ground to ensure consistent voltage reference and reliable operation.

---

### Communication Architecture
![Communication Architecture](Electrical%20Diagrams/Communication%20Architecture.png)

*Figure: Communication and signal flow between system components*

- The Raspberry Pi acts as the main controller, handling sensor processing, decision-making, and overall system logic.
- Communication interfaces are structured as follows:
  - **GPIO / PWM (low-level control signals):**
    - L298N motor driver (direction via digital output, speed via PWM)
    - SG90 servo (PWM control)
    - HC-SR04 ultrasonic sensor (TRIG/ECHO signals via GPIO)

  - **USB (high-bandwidth data communication):**
    - USB camera (image data acquisition)
    - LiDAR via USB2LDS interface (sensor data acquisition)

  - **USB (controller interfacing):**
    - OpenCR board (communication with TurtleBot base and Dynamixel motors)

## Notes
- Ensure all components share a **common ground** (OpenCR, Raspberry Pi, Sensors, and L298N)
- A voltage divider is used on the ECHO pin to safely step down the signal from 5V to 3.3V using a 2:1 resistor ratio
