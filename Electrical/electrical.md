# Electrical Documentation

## Components Used
### TurtleBot Components
- L298N Motor Driver
- SG90 Servo Motor
- RS360 DC Motor


## Connections
### SG90 Servo
- VCC → 5V rail from OpenCR board  
- GND → Common ground  
- Signal → PWM-capable GPIO pin on Raspberry Pi  

### L298N Motor Driver
- ENA (Enable Pin) → PWM-capable GPIO pin (for speed control)  
- IN1, IN2 → Digital GPIO pins (for direction control)  
- OUT1, OUT2 → Connected to the two terminals of the RS360 DC motor  

## Notes
- Ensure all components share a **common ground** (OpenCR, Raspberry Pi, and L298N).  
- PWM is used to control motor speed and servo position.
