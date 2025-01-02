
/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 https://www.arduino.cc/en/Tutorial/LibraryExamples/Sweep
*/

#include <Servo.h>

Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 0;    // variable to store the servo position


bool newData = false;        // Flag to indicate new data received

void setup() {
  Serial.begin(9600);  // Set baud rate to match Python script
  Serial.println("Arduino is ready");  // Optional: confirmation message
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  myservo.write(0);
}

void loop() {
  String receivedString = "";  // Variable to store received string
  receivedString = receiveSerialData();
  if (newData) {
    Serial.print("Received: ");
    Serial.println(receivedString);  // Send acknowledgment back to Python
//    processCommand(receivedString); // Optional: process the command
    newData = false;  // Reset the flag
  }
  if(receivedString == "0"){
    Serial.print("Servo close");
    myservo.write(0);
  }
  if(receivedString == "1"){
    Serial.print("Servo open");
    myservo.write(73);
  }
}

String receiveSerialData() {

  String receivedString = "";
  while (Serial.available() > 0) {
    char receivedChar = Serial.read();  // Read each byte
    if (receivedChar == '\n') {        // Check for end-of-line character
      newData = true;                  // Set flag when complete string is received
      break;
    } else {
      receivedString += receivedChar;  // Append character to the string
    }
  }
  return receivedString;
}

int processCommand(String command) {
  // Example: handle specific commands
  if (command == "00") {
    Serial.println("Command 00 recognized.");
    return 0;
  } else {
//    Serial.println(command);
      return 1;
  }
}
