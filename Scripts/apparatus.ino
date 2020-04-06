int rightSensorPin=7;
int leftSensorPin=8;
int middleSensorPin=9;
boolean rightVal = 0;
boolean leftVal = 0;
boolean middleVal = 0;
unsigned long leftTime;
unsigned long rightTime;
unsigned long middleTime;
unsigned long currentTime;

//whether or not a high has been detected by right mic
boolean right = false;
boolean middle = false;
boolean startMsg = false;
boolean TimeMsg = false;
boolean left = false;

void setup() {
  // put your setup code here, to run once:

  pinMode(leftSensorPin, INPUT);//left is green, yellow,blue
  pinMode(rightSensorPin, INPUT);//right is red(black),orange,brown
  pinMode(middleSensorPin, INPUT);//middle is purple,white,grey
  Serial.begin (9600);
}

void loop() {
  currentTime = micros();
  // put your main code here, to run repeatedly:

  rightVal = digitalRead(rightSensorPin);
  middleVal = digitalRead(middleSensorPin);
  leftVal = digitalRead(leftSensorPin);

  //print Start when the program starts running
  if (millis() == 200 && startMsg == false){
    Serial.println("Start");
    startMsg = true;
  }

  //detect first occurence of high input in RIGHT mic and write to rightTime
  if (right == false) {
    if (!rightVal == HIGH) {
      rightTime = currentTime;
      right = true;
    }
  }

    //detect first occurence of high input in LEFT mic and write to leftTime
  if (left == false) {
    if (!leftVal == HIGH) {
      leftTime = currentTime;
      left = true;
    }
  }
   
   //detect first occurence of high input in MIDDLE mic and write to middleTime
  if (middle == false) {
    if (!middleVal == HIGH) {
      middleTime = currentTime;
      middle = true;
    }
  }

  //print times after a given number of milliseconds
  if (millis() == 3000 && TimeMsg == false){
    Serial.println(leftTime);
    Serial.println(middleTime);
    Serial.println(rightTime);
    
    TimeMsg = true;
  }
  
}
