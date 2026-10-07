// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// FULL MOTION / IMU ML  — v006   (pairs with index-v006.html)
//
// On-device IMU data collection, training, and inference
// for education and proof of concept
//
// Input: 40 samples x 3 axes (AccelX, AccelY, AccelZ) = 120 floats per sample
// Sampling: ~40 Hz over ~1 second per capture window
//
// SD card stores: sensor recordings in class folders (.csv)
// SD card stores: weights in binary format (.bin) and .h text header
// Serial monitor and OLED output
//
// WHAT v006 CHANGES vs v005 (every change is marked "// v006:")
//   1. BUG FIX, classes 0 and 1 confused: v005 normalized every input with the std of a
//      STILL window (floored at 0.01 g) and clipped at +-5. That is a gain of about 100,
//      so any tilt or movement saturated at +-5 and "0Still" and "1Punch" became the same
//      picture for the network (a class that changes sign, like a wave, still worked).
//      v006 takes mean and std from the TRAINING windows themselves (floor MY_MIN_STD),
//      stores them in myCalib.bin and sends them in the model package, so the page and the
//      device always normalize identically. MY_NORM_FROM_DATA 0 gives the v005 behaviour back.
//   2. BUG FIX, BLE: notifications can carry only (ATT MTU - 3) bytes, 20 bytes when the phone
//      never negotiates a bigger MTU. v005 sent 163 byte frames, which were cut off, so
//      everything the XIAO SENDS (samples, models, replies) broke while everything the page
//      sends still worked (writes are split by the phone). v006 learns the MTU and splits
//      each frame into notifications that fit. Serial is unchanged.
//   3. Training on the device starts from fresh random weights (MY_TRAIN_FRESH 1, like the
//      page's default) because the normalization can change from run to run.
//   4. A baked-in header now also carries the normalization (MY_HAS_BAKED_NORM).
//   5. INFO reports MTU=<n> (0 over serial). Faster BLE connection interval is requested.
//   6. BUG FIX (also in v005): a Web Serial frame line could be read by the menu as keystrokes
//      if it arrived at the wrong moment, which lost commands at random and could even trigger
//      menu actions. myKeyAvailable() now ignores anything that starts a frame.
//
// WHAT v005 CHANGES vs v003 (only two small, additive things, both marked "// v005:").
// The page was simplified; the frames, commands, file formats and the model
// layout are exactly the same as v003, so a v003 sketch still works with the
// new page except for the two conveniences below.
//   1. INFO now ends with MC=<crc32 of the model package>. The page compares it
//      with its own model, so it can show "PHONE = XIAO" or "PHONE is not the
//      same as XIAO" instead of guessing where the current model is.
//   2. STATUS now also replies  HELLO <device name> firmware-v005  so the
//      Serial/BLE status shows which board it is talking to.
//
// WHAT v003 ADDS TO v001 (every change is marked "// v003:"):
//   1. A link to the web page over WebBLE (phone or desktop) AND over
//      Web Serial (desktop). Both carry the SAME small frames, so the page
//      can collect, pull, push, train, infer and debug without moving the
//      SD card. The SD card stays the source of truth.
//   2. Layout is compile-time #defines (see ==LAYOUT START==), with
//      static_asserts and an exact weight-file size.
//   3. A weights file of the wrong size is REFUSED (message + sketch layout).
//   4. /header/config.json is read at boot (class names only; layout is
//      compared and a WARNING is printed on mismatch). Class names become
//      folder names, so page and sketch must agree.
//   5. Every function and global is declared before use (Arduino IDE).
//   6. Small fixes: a new sample never overwrites an older one with the
//      same number, and weights/calibration can be loaded from the page.
//
// NOT TESTED ON HARDWARE. It was syntax-checked against an Arduino mock and
// its link layer was run against the page's JavaScript on a PC.
// Bench-tune MY_SIGN_X/Y/Z so Z reads about +1 g when the board lies flat.
//
// LIBRARIES
//   Seeed Arduino LSM6DS3, U8g2, and (optional) NimBLE-Arduino 2.x by h2zero.
//   If you do not want BLE, put   #define MY_USE_BLE 0   before the includes.
//   Tools -> PSRAM: OPI PSRAM.
//
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique),
//   Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
// LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// For platformio you need the U8g2 library declared in the platformio.ini file
// lib_deps = olikraus/U8g2 @ ^2.35.30
//            Seeed Arduino LSM6DS3
//            h2zero/NimBLE-Arduino @ ^2.x
// board_build.arduino.memory_type = qio_opi
//


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Globals, Declarations, Memory, Weights, Setup, Loop   ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// Optional: uncomment AFTER copying myMotionWeights.h from SD to sketch folder
// Priority order: SD weights > baked-in weights > random He-init
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS

#ifndef MY_USE_BLE
  #define MY_USE_BLE 1            // v003: 1 = WebBLE link on, 0 = no NimBLE library needed
#endif

#ifdef USE_BAKED_WEIGHTS
  #include "myMotionWeights.h"
#endif

#include <LSM6DS3.h>
#include <Wire.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <stdarg.h>               // v003
#include <U8g2lib.h>
#include "mbedtls/base64.h"       // v003
#if MY_USE_BLE
  #include <NimBLEDevice.h>       // v003
#endif


// ======================================================
// v003: LAYOUT. Everything below derives from these #defines and they are
// COMPILE-TIME on purpose (buffers are sized from them). The web page shows
// the exact lines to paste here. Change NUM_CLASSES together with the
// myClassLabels list further down.
// ==LAYOUT START==
#define NUM_CLASSES       3

#define IMU_TIMESTEPS     40
#define IMU_AXES           3      // AccelX, AccelY, AccelZ (fixed at 3 in this sketch)
#define INPUT_SIZE       (IMU_TIMESTEPS * IMU_AXES)   // 120
#define SAMPLE_INTERVAL_MS  25    // ~40 Hz => ~1 second window

// Conv1D -> MaxPool/2 -> Dense -> Dense -> Output
#define CONV1_KERNEL    5
#define CONV1_FILTERS   8
#define CONV1_OUT_STEPS (IMU_TIMESTEPS - CONV1_KERNEL + 1)   // 36
#define POOL1_STEPS     (CONV1_OUT_STEPS / 2)                // 18
#define CONV1_FLAT      (POOL1_STEPS * CONV1_FILTERS)        // 144
#define CONV1_WEIGHTS   (CONV1_KERNEL * IMU_AXES * CONV1_FILTERS)  // 120

#define DENSE1_SIZE   32
#define DENSE2_SIZE   16

#define DENSE1_WEIGHTS  (CONV1_FLAT  * DENSE1_SIZE)   // 4608
#define DENSE2_WEIGHTS  (DENSE1_SIZE * DENSE2_SIZE)   //  512
#define OUTPUT_WEIGHTS  (DENSE2_SIZE * NUM_CLASSES)   //   48

// Exact weights-file size (floats, little endian): conv w,b  dense1 w,b  dense2 w,b  output w,b
#define MY_WEIGHT_FLOATS (CONV1_WEIGHTS + CONV1_FILTERS + DENSE1_WEIGHTS + DENSE1_SIZE + \
                          DENSE2_WEIGHTS + DENSE2_SIZE + OUTPUT_WEIGHTS + NUM_CLASSES)   // 5347
#define MY_WEIGHT_BYTES  (MY_WEIGHT_FLOATS * 4)
// Package sent over the link = weights file layout + calibration mean[3] + std[3]
#define MY_PACKAGE_FLOATS (MY_WEIGHT_FLOATS + 2 * IMU_AXES)
#define MY_PACKAGE_BYTES  (MY_PACKAGE_FLOATS * 4)
// One buffer serves incoming blobs (model, sample, config.json) and outgoing ones
#define MY_BLOB_CAP ((MY_PACKAGE_BYTES) > 4200 ? (MY_PACKAGE_BYTES) : 4200)

static_assert(IMU_AXES == 3, "this sketch reads 3 accelerometer axes");
static_assert(CONV1_KERNEL >= 1, "kernel must be at least 1");
static_assert(CONV1_OUT_STEPS >= 2, "need at least 2 conv output steps for the 2x pool");
static_assert((CONV1_OUT_STEPS % 2) == 0, "IMU_TIMESTEPS - CONV1_KERNEL + 1 must be even (2x pool)");
static_assert(NUM_CLASSES >= 2 && NUM_CLASSES <= 250, "NUM_CLASSES 2..250");
static_assert(INPUT_SIZE * 4 <= MY_BLOB_CAP, "a sample must fit the blob buffer");
// ==LAYOUT END==

#ifdef USE_BAKED_WEIGHTS
  // v003: refuse a baked-in header that was made for another layout
  static_assert(sizeof(myModel_conv1_w)  / sizeof(float) == CONV1_WEIGHTS,  "baked conv1_w size differs from this layout");
  static_assert(sizeof(myModel_dense1_w) / sizeof(float) == DENSE1_WEIGHTS, "baked dense1_w size differs from this layout");
  static_assert(sizeof(myModel_dense2_w) / sizeof(float) == DENSE2_WEIGHTS, "baked dense2_w size differs from this layout");
  static_assert(sizeof(myModel_output_w) / sizeof(float) == OUTPUT_WEIGHTS, "baked output_w size differs from this layout");
#endif


// ======================================================
// v006: NORMALIZATION AND TRAINING SWITCHES
// ======================================================
#ifndef MY_NORM_FROM_DATA
  #define MY_NORM_FROM_DATA 1     // v006: 1 = mean/std of the training windows (fixes 0/1 confusion), 0 = v005 still-window calibration
#endif
#ifndef MY_TRAIN_FRESH
  #define MY_TRAIN_FRESH    1     // v006: 1 = every TRAIN starts from random weights, 0 = continue from the loaded weights
#endif
#define MY_MIN_STD 0.05f          // v006: floor for the data std (g); the page uses the same number

// ======================================================
// CONFIGURATION & ML HYPERPARAMETERS
// ======================================================
String myClassLabels[NUM_CLASSES] = {"0Still", "1Punch", "2Wave"};

const int myTotalItems = NUM_CLASSES + 2;  // classes + Train + Infer

float LEARNING_RATE  = 0.001f;
int   BATCH_SIZE     = 6;
int   TARGET_EPOCHS  = 30;
int   VALIDATION_SAMPLES = 3;   // last N samples per class held out for validation (0 = disabled)

// ======================================================
// v003: SENSOR PARITY. The page produces windows in the SAME frame:
// units of g (gravity included), order ax,ay,az, 25 ms apart.
// If your board's axes come out mirrored, flip the sign here (and tell the
// page's phone mapping the same). Bench-tune: Z should be about +1 when flat.
// Changing a sign makes older recordings incompatible: recapture them.
// ======================================================
#define MY_SIGN_X  1.0f
#define MY_SIGN_Y  1.0f
#define MY_SIGN_Z  1.0f

// ======================================================
// NORMALIZATION CONSTANTS (computed at startup by myCalibrate())
// Defaults used only if calibration is skipped.
// Z axis default mean=1.0 accounts for gravity when device is flat.
// v005 QUIRK (only used when MY_NORM_FROM_DATA is 0): std comes from a STILL window and is
// floored at 0.01, so real motion is many sigma and is clipped at +-5. THIS WAS THE BUG.
// v006 (MY_NORM_FROM_DATA 1): these two arrays hold the mean/std of the training windows;
// they are computed at the start of every TRAIN and saved to /header/myCalib.bin.
// ======================================================
float myAccelMean[IMU_AXES] = { 0.0f,  0.0f,  1.0f };
float myAccelStd [IMU_AXES] = { 1.0f,  1.0f,  1.0f };
#define CALIB_SAMPLES  80   // ~2 seconds of stationary data at 40 Hz

// ======================================================
// TOUCH INPUT SYSTEM
// ======================================================
const int myThresholdPress   = 1100;
const int myThresholdRelease =  900;

struct TouchState {
  bool          isTouching     = false;
  int           tapCount       = 0;
  unsigned long firstTapTime   = 0;
  unsigned long lastReleaseTime= 0;
  unsigned long lastCheckTime  = 0;
  const unsigned long tapWindow    = 800;
  const int           longPressTaps= 3;
  const unsigned long debounceDelay= 50;
};

TouchState myTouch;

// ======================================================
// SYSTEM LOGIC VARIABLES
// ======================================================
unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime      = 0;
const int     myTapCooldown      = 250;
int           myMenuIndex        = 1;
bool          myIsSelected       = false;
bool          myWeightsTrained   = false;
bool          mySDavailable      = false;

// ======================================================
// DEVICES
// ======================================================
LSM6DS3 myIMU(I2C_MODE, 0x6A);    // IMU object using I2C interface
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// ML WEIGHT & GRADIENT BUFFERS (PSRAM)
// ======================================================
float* myInputBuffer = nullptr;   // INPUT_SIZE floats per inference/training step

// Conv1D weights and biases
float* myConv1_w = nullptr;       // CONV1_WEIGHTS = kernel x axes x filters
float* myConv1_b = nullptr;       // CONV1_FILTERS

// Dense weights and biases
float* myDense1_w = nullptr;
float* myDense1_b = nullptr;
float* myDense2_w = nullptr;
float* myDense2_b = nullptr;
float* myOutput_w = nullptr;
float* myOutput_b = nullptr;

// Gradients
float* myConv1_w_grad  = nullptr;
float* myConv1_b_grad  = nullptr;
float* myDense1_w_grad = nullptr;
float* myDense1_b_grad = nullptr;
float* myDense2_w_grad = nullptr;
float* myDense2_b_grad = nullptr;
float* myOutput_w_grad = nullptr;
float* myOutput_b_grad = nullptr;

// Adam optimizer momentum buffers
float* myConv1_w_m  = nullptr;  float* myConv1_w_v  = nullptr;
float* myConv1_b_m  = nullptr;  float* myConv1_b_v  = nullptr;
float* myDense1_w_m = nullptr;  float* myDense1_w_v = nullptr;
float* myDense1_b_m = nullptr;  float* myDense1_b_v = nullptr;
float* myDense2_w_m = nullptr;  float* myDense2_w_v = nullptr;
float* myDense2_b_m = nullptr;  float* myDense2_b_v = nullptr;
float* myOutput_w_m = nullptr;  float* myOutput_w_v = nullptr;
float* myOutput_b_m = nullptr;  float* myOutput_b_v = nullptr;

// Forward-pass activation buffers
float* myConv1_output  = nullptr;   // CONV1_OUT_STEPS x CONV1_FILTERS (pre-pool)
float* myPool1_output  = nullptr;   // POOL1_STEPS     x CONV1_FILTERS = CONV1_FLAT (post-pool)
float* myDense1_output = nullptr;   // DENSE1_SIZE
float* myDense2_output = nullptr;   // DENSE2_SIZE
float* myFinal_output  = nullptr;   // NUM_CLASSES  (softmax probabilities)

// Backward-pass delta buffers
float* myOutput_delta = nullptr;   // NUM_CLASSES
float* myDense2_delta = nullptr;   // DENSE2_SIZE
float* myDense1_delta = nullptr;   // DENSE1_SIZE
float* myPool1_delta  = nullptr;   // CONV1_FLAT
float* myConv1_delta  = nullptr;   // CONV1_OUT_STEPS x CONV1_FILTERS

// Adam step counter (QUIRK kept: it counts once per parameter ARRAY update)
int myAdamStep = 0;

struct TrainingItem {
  String path;
  int    label;
};
std::vector<TrainingItem> myTrainingData;

// ======================================================
// v003: LINK STATE (BLE + Web Serial share the same frames)
// Frame = [type][payload]. Types: 'H' blob header, 'D' blob chunk,
// 'A' ack, 'T' text command (page -> device), 'R' text reply (device -> page).
// BLE carries a frame per write/notify. Serial carries "@B <base64(frame)>\n".
// A "blob" (model, sample, config.json) is sent as H, then chunks of
// MY_CHUNK_DATA bytes, acked every MY_WIN chunks with the next expected chunk.
// ==LINK VARS START==
#define MY_DEVICE_NAME  "XIAO-Motion-01"
#define MY_SVC_UUID     "7e600001-b2c3-5d4e-af60-9b3c7d8eaf20"
#define MY_CMD_UUID     "7e600002-b2c3-5d4e-af60-9b3c7d8eaf20"   // page -> device, write with response
#define MY_EVT_UUID     "7e600003-b2c3-5d4e-af60-9b3c7d8eaf20"   // device -> page, notify

#define MY_CHUNK_DATA   160
#define MY_FRAME_MAX    (MY_CHUNK_DATA + 8)
#define MY_WIN          4
#define MY_ACK_TIMEOUT_MS 2500
#define MY_QN           12
#define MY_F_HEAD  'H'
#define MY_F_DATA  'D'
#define MY_F_ACK   'A'
#define MY_F_TEXT  'T'
#define MY_F_RESP  'R'

struct MyFrame {
  uint8_t len;
  uint8_t via;                  // 1 = BLE, 2 = Serial
  uint8_t d[MY_FRAME_MAX];
};
MyFrame myQ[MY_QN];             // BLE callback -> loop() queue
volatile uint8_t myQHead = 0;
volatile uint8_t myQTail = 0;

volatile uint8_t myReplyVia = 0;        // where replies go: 0 nowhere, 1 BLE, 2 Serial
volatile bool    myBleConnected = false;
bool             myLinkDebugOn = false;
unsigned long    myDebugLastMs = 0;
unsigned long    myLastHbMs = 0;
volatile bool    myBusy = false;
volatile bool    myStopRequested = false;
uint8_t*         myBlobBuf = nullptr;   // MY_BLOB_CAP bytes in PSRAM

volatile bool    myRxActive = false;    // a blob is arriving
uint8_t          myRxKind = 0;
uint8_t          myRxId = 0;
uint32_t         myRxTotal = 0;
uint32_t         myRxGot = 0;
uint32_t         myRxCrc = 0;
uint16_t         myRxNext = 0;
unsigned long    myRxLastMs = 0;

volatile bool    mySending = false;     // inside mySendBlob(): once its ack arrives, stop reading so the NEXT command is handled after this one finishes
volatile uint16_t myTxAckNext = 0xFFFF; // last ack from the page (0xFFFF = none yet)
volatile uint8_t  myTxAckStatus = 0;    // 0 ok, 1 crc error, 2 abort

char             mySerLine[240];        // one "@B ..." line from Web Serial
int              mySerLen = 0;
// ==LINK VARS END==

#if MY_USE_BLE
volatile uint16_t     myBleMtu    = 23;     // v006: negotiated ATT MTU (23 until the phone asks for more)
NimBLEServer*         myBleServer = nullptr;
NimBLECharacteristic* myCmdChar   = nullptr;
NimBLECharacteristic* myEvtChar   = nullptr;
#endif

#define MY_NAME_MAX 32
bool myLinkWasDebug = false;


// ======================================================
// UTILITY FUNCTIONS (defined before use)
// ======================================================
inline float myClip(float v, float mn=-100, float mx=100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}

inline float myLeakyRelu(float x)      { return x > 0 ? x : 0.1f * x; }
inline float myLeakyReluDeriv(float x) { return x > 0 ? 1.0f : 0.1f; }


// ======================================================
// v003: FORWARD DECLARATIONS of every function (Arduino IDE friendly).
// ======================================================
void  mySoftmax(float* x, int size);
void  myNormalizeInput(float* buf);
void  myReadAccel(float* v);
int   myReadTouch();
void  myResetTouchState();
void  myUpdateTouchState();
int   myCheckTouchInput();
void  myCheckTouchBackground();
void  myAllocateMemory();
void  myPrintLayout();
void  myExportHeader();
bool  myLoadWeights();
void  mySaveWeights();
void  mySaveCalib();
void  myCalibrate(bool force);
bool  myLoadCalibFromSD();                              // v006
void  myComputeNormFromData(const std::vector<TrainingItem>& items, float* buf);   // v006
void  myInitWeights();                                  // v006
void  myResetAdam();                                    // v006
void  myPackToBuf(float* out);
bool  myPackFromBuf(const float* in);
int   myCfgFindKey(const char* t, const char* key);
bool  myCfgInt(const char* t, const char* key, int* out);
int   myCfgClasses(const char* t, char names[][MY_NAME_MAX], int maxN);
void  myApplyConfig(const char* t);
void  myLoadConfigFromSD();
int   myCountSamples(int classIdx);
bool  myNewSamplePath(int classIdx, String* out);
bool  myNthSamplePath(int classIdx, int n, String* out);
void  myCaptureWindow(float* w, bool echo);
bool  myWriteSample(int classIdx, const float* w, String* pathOut);
bool  myCaptureSample(int classIdx);
void  myActionCollect(int classIdx);
void  myConv1DForward(float* input);
void  myPool1Forward();
void  myDenseForward(float* input, int inSize, float* w, float* b, float* output, int outSize, bool applyActivation);
void  myForwardPass(float* input);
float myComputeLoss(int label);
void  myAdamUpdate(float* w, float* grad, float* m, float* v, int size, float lr);
void  myZeroGradients();
void  myBackwardPass(float* input, int label);
bool  myReadSampleCsv(const char* path, float* buf);
bool  myLoadSampleFromFile(const char* path, float* buf);
bool  myTrainCore();
void  myActionTrain();
void  myActionInfer();
void  myResetMenuState();
void  myDrawMenu();
void  myExecuteMenuItem(int idx);
void  myHandleMenuNavigation();
bool  myKeyAvailable();
char  myKeyRead();
// link layer
uint32_t myCrc32(const uint8_t* d, size_t n);
uint32_t myModelCrc();                  // v005
void  myPut32(uint8_t* p, uint32_t v);
uint32_t myGet32(const uint8_t* p);
bool  mySendFrame(const uint8_t* d, size_t n);
void  myReply(const char* fmt, ...);
void  myAck(uint16_t next, uint8_t status);
bool  mySendBlob(uint8_t kind, uint8_t id, const uint8_t* data, uint32_t total);
void  myOnHead(const uint8_t* d);
void  myOnData(const uint8_t* d, size_t n);
void  myHandleFrame(const uint8_t* d, size_t n);
void  myQPush(const uint8_t* d, size_t n, uint8_t via);
void  myPollSerialFrames();
void  myPumpIncoming();
// commands (need the sketch's data, so they are outside the link layer)
void  myHandleCommand(char* s);
void  myDispatchBlob(uint8_t kind, uint8_t id, uint32_t total);
void  myImportPackage();
void  myStoreSample(int classIdx);
void  myStoreConfig(uint32_t total);
void  myReplyInfo();
void  myReplyCal();
void  myLinkHeartbeat();
#if MY_USE_BLE
void  myStartBle();
bool  mySendNotify(const uint8_t* p, size_t n);         // v006
#endif


// ======================================================
// INPUT HELPERS
// ======================================================
void mySoftmax(float* x, int size) {
  float maxVal = x[0];
  for (int i = 1; i < size; i++) if (x[i] > maxVal) maxVal = x[i];
  float sum = 0;
  for (int i = 0; i < size; i++) { x[i] = exp(x[i] - maxVal); sum += x[i]; }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

// Normalize one full input window using per-axis mean/std
// Input buffer layout: [t0_ax, t0_ay, t0_az, t1_ax, t1_ay, t1_az, ...]
void myNormalizeInput(float* buf) {
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    for (int a = 0; a < IMU_AXES; a++) {
      int idx = t * IMU_AXES + a;
      buf[idx] = (buf[idx] - myAccelMean[a]) / (myAccelStd[a] + 1e-8f);
      buf[idx] = myClip(buf[idx], -5.0f, 5.0f);
    }
  }
}

// v003: ONE place that reads the IMU, in g, with the sign settings above.
void myReadAccel(float* v) {
  v[0] = MY_SIGN_X * myIMU.readFloatAccelX();
  v[1] = MY_SIGN_Y * myIMU.readFloatAccelY();
  v[2] = MY_SIGN_Z * myIMU.readFloatAccelZ();
}

// v003: all single-key input goes through these, so "@B ..." frame lines
// from the page are consumed first and never reach the menu.
bool myKeyAvailable() {
  myPollSerialFrames();
  // v006 BUG FIX (present in v005): a frame line that arrived between the poll above and a plain
  // Serial.available() test was read by the menu one byte at a time ("@", "B", " " ...). Its base64
  // text contains digits, 't' and 'l', which are menu commands. Report a key only if the next byte
  // is not the start of a frame and no frame line is half read.
  if (mySerLen > 0) return false;
  int myNext = Serial.peek();
  return myNext >= 0 && myNext != '@';
}
char myKeyRead() {
  return (char)Serial.read();
}


// ======================================================
// TOUCH INPUT FUNCTIONS
// ======================================================
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}

void myResetTouchState() {
  myTouch.isTouching      = false;
  myTouch.tapCount        = 0;
  myTouch.firstTapTime    = 0;
  myTouch.lastReleaseTime = 0;
  myTouch.lastCheckTime   = 0;
}

void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;

  int  val         = myReadTouch();
  bool touchActive = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);

  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount = 1;
      myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }
  if (!touchActive && myTouch.isTouching) {
    myTouch.isTouching      = false;
    myTouch.lastReleaseTime = now;
  }
}

// Returns: 0=no action, 1=tap, 2=long press (3+ taps)
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
      int count  = myTouch.tapCount;
      myResetTouchState();
      Serial.printf(result == 2 ? "LONG PRESS (%d taps)\n" : "TAP (%d tap%s)\n",
                    count, count > 1 ? "s" : "");
      return result;
    }
  }
  return 0;
}

// Non-blocking touch state update for use inside heavy computation loops
void myCheckTouchBackground() {
  myUpdateTouchState();
}


// ======================================================
// MEMORY ALLOCATION
// ======================================================
void myAllocateMemory() {
  if (myInputBuffer != nullptr) return;
  Serial.println("\n=== Allocating Memory ===");

  myInputBuffer  = (float*)ps_malloc(INPUT_SIZE     * sizeof(float));
  myBlobBuf      = (uint8_t*)ps_malloc(MY_BLOB_CAP);          // v003

  // Conv1D
  myConv1_w      = (float*)ps_malloc(CONV1_WEIGHTS  * sizeof(float));
  myConv1_b      = (float*)ps_malloc(CONV1_FILTERS  * sizeof(float));

  // Dense
  myDense1_w     = (float*)ps_malloc(DENSE1_WEIGHTS * sizeof(float));
  myDense1_b     = (float*)ps_malloc(DENSE1_SIZE    * sizeof(float));
  myDense2_w     = (float*)ps_malloc(DENSE2_WEIGHTS * sizeof(float));
  myDense2_b     = (float*)ps_malloc(DENSE2_SIZE    * sizeof(float));
  myOutput_w     = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b     = (float*)ps_malloc(NUM_CLASSES    * sizeof(float));

  // Gradients
  myConv1_w_grad  = (float*)ps_malloc(CONV1_WEIGHTS  * sizeof(float));
  myConv1_b_grad  = (float*)ps_malloc(CONV1_FILTERS  * sizeof(float));
  myDense1_w_grad = (float*)ps_malloc(DENSE1_WEIGHTS * sizeof(float));
  myDense1_b_grad = (float*)ps_malloc(DENSE1_SIZE    * sizeof(float));
  myDense2_w_grad = (float*)ps_malloc(DENSE2_WEIGHTS * sizeof(float));
  myDense2_b_grad = (float*)ps_malloc(DENSE2_SIZE    * sizeof(float));
  myOutput_w_grad = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b_grad = (float*)ps_malloc(NUM_CLASSES    * sizeof(float));

  // Adam buffers (zero-initialised)
  myConv1_w_m  = (float*)ps_calloc(CONV1_WEIGHTS,  sizeof(float));
  myConv1_w_v  = (float*)ps_calloc(CONV1_WEIGHTS,  sizeof(float));
  myConv1_b_m  = (float*)ps_calloc(CONV1_FILTERS,  sizeof(float));
  myConv1_b_v  = (float*)ps_calloc(CONV1_FILTERS,  sizeof(float));
  myDense1_w_m = (float*)ps_calloc(DENSE1_WEIGHTS, sizeof(float));
  myDense1_w_v = (float*)ps_calloc(DENSE1_WEIGHTS, sizeof(float));
  myDense1_b_m = (float*)ps_calloc(DENSE1_SIZE,    sizeof(float));
  myDense1_b_v = (float*)ps_calloc(DENSE1_SIZE,    sizeof(float));
  myDense2_w_m = (float*)ps_calloc(DENSE2_WEIGHTS, sizeof(float));
  myDense2_w_v = (float*)ps_calloc(DENSE2_WEIGHTS, sizeof(float));
  myDense2_b_m = (float*)ps_calloc(DENSE2_SIZE,    sizeof(float));
  myDense2_b_v = (float*)ps_calloc(DENSE2_SIZE,    sizeof(float));
  myOutput_w_m = (float*)ps_calloc(OUTPUT_WEIGHTS,  sizeof(float));
  myOutput_w_v = (float*)ps_calloc(OUTPUT_WEIGHTS,  sizeof(float));
  myOutput_b_m = (float*)ps_calloc(NUM_CLASSES,     sizeof(float));
  myOutput_b_v = (float*)ps_calloc(NUM_CLASSES,     sizeof(float));

  // Forward-pass buffers
  myConv1_output  = (float*)ps_malloc(CONV1_OUT_STEPS * CONV1_FILTERS * sizeof(float));
  myPool1_output  = (float*)ps_malloc(CONV1_FLAT       * sizeof(float));
  myDense1_output = (float*)ps_malloc(DENSE1_SIZE      * sizeof(float));
  myDense2_output = (float*)ps_malloc(DENSE2_SIZE      * sizeof(float));
  myFinal_output  = (float*)ps_malloc(NUM_CLASSES      * sizeof(float));

  // Backward-pass buffers
  myOutput_delta = (float*)ps_malloc(NUM_CLASSES                      * sizeof(float));
  myDense2_delta = (float*)ps_malloc(DENSE2_SIZE                      * sizeof(float));
  myDense1_delta = (float*)ps_malloc(DENSE1_SIZE                      * sizeof(float));
  myPool1_delta  = (float*)ps_malloc(CONV1_FLAT                       * sizeof(float));
  myConv1_delta  = (float*)ps_malloc(CONV1_OUT_STEPS * CONV1_FILTERS  * sizeof(float));

  if (!myInputBuffer || !myBlobBuf || !myConv1_w || !myDense1_w || !myDense2_w || !myOutput_w ||
      !myConv1_output || !myPool1_output || !myDense1_output || !myFinal_output) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while (1) { delay(1000); }
  }

  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  myInitWeights();                                     // v006: was inline here
}


// v006: the He initialization, moved out of myAllocateMemory() so every TRAIN can start fresh.
void myInitWeights() {
  // He initialization (the constants derive from the layout #defines)
  // Conv1D: fan-in = kernel x axes
  float c1std = sqrt(2.0f / (CONV1_KERNEL * IMU_AXES));
  for (int i = 0; i < CONV1_WEIGHTS; i++)
    myConv1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c1std;
  for (int i = 0; i < CONV1_FILTERS; i++) myConv1_b[i] = 0;

  // Dense1: fan-in = CONV1_FLAT
  float d1std = sqrt(2.0f / CONV1_FLAT);
  for (int i = 0; i < DENSE1_WEIGHTS; i++)
    myDense1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * d1std;
  for (int i = 0; i < DENSE1_SIZE; i++) myDense1_b[i] = 0;

  float d2std = sqrt(2.0f / DENSE1_SIZE);
  for (int i = 0; i < DENSE2_WEIGHTS; i++)
    myDense2_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * d2std;
  for (int i = 0; i < DENSE2_SIZE; i++) myDense2_b[i] = 0;

  float ostd = sqrt(2.0f / DENSE2_SIZE);
  for (int i = 0; i < OUTPUT_WEIGHTS; i++)
    myOutput_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * ostd;
  for (int i = 0; i < NUM_CLASSES; i++) myOutput_b[i] = 0;

  Serial.println("He-init random weights set");
}

// v006: forget the Adam history (a new training run starts from zero)
void myResetAdam() {
  memset(myConv1_w_m,  0, CONV1_WEIGHTS  * sizeof(float));  memset(myConv1_w_v,  0, CONV1_WEIGHTS  * sizeof(float));
  memset(myConv1_b_m,  0, CONV1_FILTERS  * sizeof(float));  memset(myConv1_b_v,  0, CONV1_FILTERS  * sizeof(float));
  memset(myDense1_w_m, 0, DENSE1_WEIGHTS * sizeof(float));  memset(myDense1_w_v, 0, DENSE1_WEIGHTS * sizeof(float));
  memset(myDense1_b_m, 0, DENSE1_SIZE    * sizeof(float));  memset(myDense1_b_v, 0, DENSE1_SIZE    * sizeof(float));
  memset(myDense2_w_m, 0, DENSE2_WEIGHTS * sizeof(float));  memset(myDense2_w_v, 0, DENSE2_WEIGHTS * sizeof(float));
  memset(myDense2_b_m, 0, DENSE2_SIZE    * sizeof(float));  memset(myDense2_b_v, 0, DENSE2_SIZE    * sizeof(float));
  memset(myOutput_w_m, 0, OUTPUT_WEIGHTS * sizeof(float));  memset(myOutput_w_v, 0, OUTPUT_WEIGHTS * sizeof(float));
  memset(myOutput_b_m, 0, NUM_CLASSES    * sizeof(float));  memset(myOutput_b_v, 0, NUM_CLASSES    * sizeof(float));
  myAdamStep = 0;
}

// v003: prints what THIS sketch was compiled for (used in refusal messages)
void myPrintLayout() {
  Serial.printf("Sketch layout: NUM_CLASSES=%d IMU_TIMESTEPS=%d IMU_AXES=%d CONV1_KERNEL=%d CONV1_FILTERS=%d "
                "DENSE1_SIZE=%d DENSE2_SIZE=%d SAMPLE_INTERVAL_MS=%d -> %d weight floats (%d bytes)\n",
                NUM_CLASSES, IMU_TIMESTEPS, IMU_AXES, CONV1_KERNEL, CONV1_FILTERS,
                DENSE1_SIZE, DENSE2_SIZE, SAMPLE_INTERVAL_MS, MY_WEIGHT_FLOATS, MY_WEIGHT_BYTES);
}


// ======================================================
// WEIGHT SAVE / LOAD / EXPORT
// ======================================================
void myExportHeader() {
  if (!mySDavailable) { Serial.println("No SD card - cannot export header"); return; }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myMotionWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MOTION_MODEL_H\n#define MY_MOTION_MODEL_H");
  file.println("// Uncomment in main sketch:  #define USE_BAKED_WEIGHTS");
  file.printf( "// #define NUM_CLASSES %d\n", NUM_CLASSES);
  file.print("// String myClassLabels[NUM_CLASSES] = {");
  for (int i = 0; i < NUM_CLASSES; i++) {
    file.printf("\"%s\"", myClassLabels[i].c_str());
    if (i < NUM_CLASSES - 1) file.print(", ");
  }
  file.println("};");

  auto myDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = { ", name);
    for (int i = 0; i < size; i++) {
      file.print(data[i], 6); file.print("f");
      if (i < size - 1) file.print(", ");
      if ((i + 1) % 8 == 0) file.println();
    }
    file.println(" };");
  };
  myDump("myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDump("myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDump("myModel_dense1_w", myDense1_w, DENSE1_WEIGHTS);
  myDump("myModel_dense1_b", myDense1_b, DENSE1_SIZE);
  myDump("myModel_dense2_w", myDense2_w, DENSE2_WEIGHTS);
  myDump("myModel_dense2_b", myDense2_b, DENSE2_SIZE);
  myDump("myModel_output_w", myOutput_w, OUTPUT_WEIGHTS);
  myDump("myModel_output_b", myOutput_b, NUM_CLASSES);
  file.println("#define MY_HAS_BAKED_NORM   // v006: the normalization that goes with these weights");   // v006
  myDump("myModel_norm_mean", myAccelMean, IMU_AXES);                                                // v006
  myDump("myModel_norm_std",  myAccelStd,  IMU_AXES);                                                // v006
  file.println("#endif");
  file.close();
  Serial.println("Header exported to /header/myMotionWeights.h");
}

bool myLoadWeights() {
  if (!mySDavailable) { Serial.println("No SD - skipping weight load"); return false; }
  if (!SD.exists("/header/myMotionWeights.bin")) { Serial.println("No weights file found"); return false; }
  Serial.println("Loading weights from SD...");
  File f = SD.open("/header/myMotionWeights.bin", FILE_READ);
  if (!f) return false;
  // v003: refuse a file made for another layout instead of loading garbage
  if ((size_t)f.size() != (size_t)MY_WEIGHT_BYTES) {
    Serial.printf("REFUSED: myMotionWeights.bin is %u bytes but this sketch needs %u bytes.\n",
                  (unsigned)f.size(), (unsigned)MY_WEIGHT_BYTES);
    myPrintLayout();
    f.close();
    return false;
  }
  f.read((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
  f.read((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
  f.read((uint8_t*)myDense1_w, DENSE1_WEIGHTS * 4);
  f.read((uint8_t*)myDense1_b, DENSE1_SIZE    * 4);
  f.read((uint8_t*)myDense2_w, DENSE2_WEIGHTS * 4);
  f.read((uint8_t*)myDense2_b, DENSE2_SIZE    * 4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
  f.read((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
  f.close();
  Serial.println("Weights loaded successfully");
  myWeightsTrained = true;
  return true;
}

void mySaveWeights() {
  if (!mySDavailable) { Serial.println("No SD - cannot save weights"); return; }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myMotionWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
    f.write((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
    f.write((uint8_t*)myDense1_w, DENSE1_WEIGHTS * 4);
    f.write((uint8_t*)myDense1_b, DENSE1_SIZE    * 4);
    f.write((uint8_t*)myDense2_w, DENSE2_WEIGHTS * 4);
    f.write((uint8_t*)myDense2_b, DENSE2_SIZE    * 4);
    f.write((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
    f.write((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
    f.close();
    Serial.println("Weights saved to SD");
  }
  myExportHeader();
}

// v003: split out of myCalibrate() so a model pushed from the page can save its calibration too
void mySaveCalib() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myCalib.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myAccelMean, IMU_AXES * 4);
    f.write((uint8_t*)myAccelStd,  IMU_AXES * 4);
    f.close();
    Serial.println("Calibration saved to SD");
  }
}

// v003: weights in file order, then calibration. Used for the link package.
void myPackToBuf(float* out) {
  float* src[8]  = { myConv1_w, myConv1_b, myDense1_w, myDense1_b, myDense2_w, myDense2_b, myOutput_w, myOutput_b };
  int    cnt[8]  = { CONV1_WEIGHTS, CONV1_FILTERS, DENSE1_WEIGHTS, DENSE1_SIZE, DENSE2_WEIGHTS, DENSE2_SIZE, OUTPUT_WEIGHTS, NUM_CLASSES };
  int o = 0;
  for (int b = 0; b < 8; b++) { memcpy(out + o, src[b], cnt[b] * 4); o += cnt[b]; }
  memcpy(out + o, myAccelMean, IMU_AXES * 4); o += IMU_AXES;
  memcpy(out + o, myAccelStd,  IMU_AXES * 4);
}

// v003: returns false (and changes nothing) if the package holds NaN or Infinity
bool myPackFromBuf(const float* in) {
  for (int i = 0; i < MY_PACKAGE_FLOATS; i++) if (isnan(in[i]) || isinf(in[i])) return false;
  float* dst[8]  = { myConv1_w, myConv1_b, myDense1_w, myDense1_b, myDense2_w, myDense2_b, myOutput_w, myOutput_b };
  int    cnt[8]  = { CONV1_WEIGHTS, CONV1_FILTERS, DENSE1_WEIGHTS, DENSE1_SIZE, DENSE2_WEIGHTS, DENSE2_SIZE, OUTPUT_WEIGHTS, NUM_CLASSES };
  int o = 0;
  for (int b = 0; b < 8; b++) { memcpy(dst[b], in + o, cnt[b] * 4); o += cnt[b]; }
  memcpy(myAccelMean, in + o, IMU_AXES * 4); o += IMU_AXES;
  memcpy(myAccelStd,  in + o, IMU_AXES * 4);
  return true;
}


// ======================================================
// v003: CONFIG.JSON  (tiny hand-written parser, no JSON library)
// Only "classes" is used. The layout numbers are only COMPARED (WARNING).
// ==CFG PARSE START==
// Returns the index just after  "key" :  (skipping spaces), or -1 if the key is missing.
int myCfgFindKey(const char* t, const char* key) {
  char pat[32];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char* p = strstr(t, pat);
  if (!p) return -1;
  p += strlen(pat);
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  if (*p != ':') return -1;
  p++;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  return (int)(p - t);
}

bool myCfgInt(const char* t, const char* key, int* out) {
  int i = myCfgFindKey(t, key);
  if (i < 0) return false;
  *out = atoi(t + i);
  return true;
}

// Reads  "classes": ["a","b",...]  Returns how many strings the list holds
// (stores at most maxN of them), or -1 if there is no list.
int myCfgClasses(const char* t, char names[][MY_NAME_MAX], int maxN) {
  int i = myCfgFindKey(t, "classes");
  if (i < 0 || t[i] != '[') return -1;
  const char* p = t + i + 1;
  int count = 0;
  while (*p && *p != ']') {
    if (*p == '"') {
      p++;
      int n = 0;
      char tmp[MY_NAME_MAX];
      while (*p && *p != '"') { if (n < MY_NAME_MAX - 1) tmp[n++] = *p; p++; }
      tmp[n] = 0;
      if (*p == '"') p++;
      if (count < maxN) strcpy(names[count], tmp);
      count++;
    } else {
      p++;
    }
  }
  return count;
}
// ==CFG PARSE END==

void myApplyConfig(const char* t) {
  char names[NUM_CLASSES + 1][MY_NAME_MAX];
  int n = myCfgClasses(t, names, NUM_CLASSES + 1);
  if (n == NUM_CLASSES) {
    for (int i = 0; i < NUM_CLASSES; i++) myClassLabels[i] = String(names[i]);
    Serial.println("config.json: class labels loaded");
  } else if (n >= 0) {
    Serial.printf("config.json: lists %d classes but this sketch has NUM_CLASSES=%d - keeping compiled labels\n", n, NUM_CLASSES);
  } else {
    Serial.println("config.json: no \"classes\" list - keeping compiled labels");
  }
  int v;
  if (myCfgInt(t, "timesteps",   &v) && v != IMU_TIMESTEPS)      Serial.printf("WARNING: config timesteps=%d but sketch IMU_TIMESTEPS=%d\n", v, IMU_TIMESTEPS);
  if (myCfgInt(t, "axes",        &v) && v != IMU_AXES)           Serial.printf("WARNING: config axes=%d but sketch IMU_AXES=%d\n", v, IMU_AXES);
  if (myCfgInt(t, "kernel",      &v) && v != CONV1_KERNEL)       Serial.printf("WARNING: config kernel=%d but sketch CONV1_KERNEL=%d\n", v, CONV1_KERNEL);
  if (myCfgInt(t, "filters",     &v) && v != CONV1_FILTERS)      Serial.printf("WARNING: config filters=%d but sketch CONV1_FILTERS=%d\n", v, CONV1_FILTERS);
  if (myCfgInt(t, "dense1",      &v) && v != DENSE1_SIZE)        Serial.printf("WARNING: config dense1=%d but sketch DENSE1_SIZE=%d\n", v, DENSE1_SIZE);
  if (myCfgInt(t, "dense2",      &v) && v != DENSE2_SIZE)        Serial.printf("WARNING: config dense2=%d but sketch DENSE2_SIZE=%d\n", v, DENSE2_SIZE);
  if (myCfgInt(t, "interval_ms", &v) && v != SAMPLE_INTERVAL_MS) Serial.printf("WARNING: config interval_ms=%d but sketch SAMPLE_INTERVAL_MS=%d\n", v, SAMPLE_INTERVAL_MS);
  if (myCfgInt(t, "input_size",  &v) && v != INPUT_SIZE)         Serial.printf("WARNING: config input_size=%d but sketch INPUT_SIZE=%d\n", v, INPUT_SIZE);
}

void myLoadConfigFromSD() {
  if (!mySDavailable || !myBlobBuf) return;
  if (!SD.exists("/header/config.json")) { Serial.println("No /header/config.json - using compiled class labels"); return; }
  File f = SD.open("/header/config.json", FILE_READ);
  if (!f) return;
  size_t n = f.size();
  if (n == 0 || n > 4096) { Serial.println("config.json: empty or larger than 4 KB - ignored"); f.close(); return; }
  f.read(myBlobBuf, n);
  f.close();
  myBlobBuf[n] = 0;
  myApplyConfig((const char*)myBlobBuf);
}


// ======================================================
// CALIBRATION  (run once at startup, device stationary)
// Collects CALIB_SAMPLES readings and computes per-axis mean and std.
// Saves result to SD so subsequent boots skip the wait.
// v003: force=true recalibrates even if a saved file exists (page button).
// ======================================================
void myCalibrate(bool force) {
  // Try loading from SD first
  if (!force && mySDavailable && SD.exists("/header/myCalib.bin")) {
    File f = SD.open("/header/myCalib.bin", FILE_READ);
    if (f && f.size() == IMU_AXES * 2 * 4) {
      f.read((uint8_t*)myAccelMean, IMU_AXES * 4);
      f.read((uint8_t*)myAccelStd,  IMU_AXES * 4);
      f.close();
      Serial.printf("Calibration loaded: mean=%.3f,%.3f,%.3f  std=%.3f,%.3f,%.3f\n",
                    myAccelMean[0], myAccelMean[1], myAccelMean[2],
                    myAccelStd[0],  myAccelStd[1],  myAccelStd[2]);
      return;
    }
    if (f) f.close();
  }

  Serial.println("Calibrating IMU - keep device stationary...");
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 10, "Calibrating...");
    u8g2.drawStr(0, 22, "Keep still!");
  } while (u8g2.nextPage());
  delay(500);

  float sum[IMU_AXES]  = {0, 0, 0};
  float sum2[IMU_AXES] = {0, 0, 0};

  for (int i = 0; i < CALIB_SAMPLES; i++) {
    float v[IMU_AXES];
    myReadAccel(v);                                    // v003
    for (int a = 0; a < IMU_AXES; a++) { sum[a] += v[a]; sum2[a] += v[a] * v[a]; }
    delay(SAMPLE_INTERVAL_MS);
  }

  for (int a = 0; a < IMU_AXES; a++) {
    myAccelMean[a] = sum[a] / CALIB_SAMPLES;
    float var = (sum2[a] / CALIB_SAMPLES) - (myAccelMean[a] * myAccelMean[a]);
    float sd = sqrt(var);                      // v003: written without max() so it compiles whether max is a macro or a template
    myAccelStd[a]  = (sd > 0.01f) ? sd : 0.01f;   // floor at 0.01 to avoid div/0
  }

  Serial.printf("Calibration done: mean=%.3f,%.3f,%.3f  std=%.3f,%.3f,%.3f\n",
                myAccelMean[0], myAccelMean[1], myAccelMean[2],
                myAccelStd[0],  myAccelStd[1],  myAccelStd[2]);

  mySaveCalib();                                       // v003 (was inline)
}


// v006: load /header/myCalib.bin if it is there (no waiting, no IMU). Returns true if loaded.
bool myLoadCalibFromSD() {
  if (!mySDavailable || !SD.exists("/header/myCalib.bin")) return false;
  File f = SD.open("/header/myCalib.bin", FILE_READ);
  if (!f) return false;
  if (f.size() != IMU_AXES * 2 * 4) { f.close(); return false; }
  f.read((uint8_t*)myAccelMean, IMU_AXES * 4);
  f.read((uint8_t*)myAccelStd,  IMU_AXES * 4);
  f.close();
  Serial.printf("Normalization loaded: mean=%.3f,%.3f,%.3f  std=%.3f,%.3f,%.3f\n",
                myAccelMean[0], myAccelMean[1], myAccelMean[2], myAccelStd[0], myAccelStd[1], myAccelStd[2]);
  return true;
}

// v006: mean and std per axis over every timestep of every TRAINING window (raw g, no clipping).
// This replaces the still-window std that made Still and Punch look identical.
// buf = scratch space of INPUT_SIZE floats.
void myComputeNormFromData(const std::vector<TrainingItem>& items, float* buf) {
  double sum[IMU_AXES]  = {0, 0, 0};
  double sum2[IMU_AXES] = {0, 0, 0};
  long   n = 0;                                        // timesteps counted per axis
  for (size_t i = 0; i < items.size(); i++) {
    if (!myReadSampleCsv(items[i].path.c_str(), buf)) continue;
    for (int t = 0; t < IMU_TIMESTEPS; t++) {
      for (int a = 0; a < IMU_AXES; a++) {
        double v = buf[t * IMU_AXES + a];
        sum[a] += v; sum2[a] += v * v;
      }
      n++;
    }
  }
  if (n == 0) { Serial.println("Normalization: no readable windows - keeping the old values"); return; }
  for (int a = 0; a < IMU_AXES; a++) {
    double mean = sum[a] / n;
    double var  = sum2[a] / n - mean * mean;
    double sd   = (var > 0) ? sqrt(var) : 0;
    myAccelMean[a] = (float)mean;
    myAccelStd[a]  = (sd > MY_MIN_STD) ? (float)sd : MY_MIN_STD;
  }
  Serial.printf("Normalization from %ld timesteps of training data: mean=%.3f,%.3f,%.3f  std=%.3f,%.3f,%.3f\n",
                n, myAccelMean[0], myAccelMean[1], myAccelMean[2], myAccelStd[0], myAccelStd[1], myAccelStd[2]);
}


// ======================================================
// SETUP AND LOOP
// ======================================================
void setup() {
  Serial.setRxBufferSize(1024);   // v003: room for a few "@B ..." frame lines (comment out if your core has no such call)
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);

  Serial.println("\n=== XIAO ESP32-S3 Motion ML System v006 Starting ===");
  Serial.printf("Free heap:  %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  myPrintLayout();                                     // v003

  pinMode(A0, INPUT);
  u8g2.begin();

  // SD card init
  pinMode(21, OUTPUT);
  digitalWrite(21, HIGH);
  delay(100);
  Serial.println("Checking SD card...");
  SPI.begin();
  SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card - continuing without it");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
  }

  // IMU init
  if (myIMU.begin() != 0) {
    Serial.println("ERROR: IMU initialization failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "IMU ERROR!"); } while (u8g2.nextPage());
    while (1) { delay(1000); }
  }
  Serial.println("IMU initialized successfully");
#if MY_NORM_FROM_DATA
  Serial.println("Normalization: from the training data (v006). No still calibration needed.");   // v006
#else
  myCalibrate(false);                                  // v005 behaviour
#endif

  myAllocateMemory();
  myLoadConfigFromSD();                                // v003 (needs the blob buffer)

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
  memcpy(myDense1_w, myModel_dense1_w, DENSE1_WEIGHTS * sizeof(float));
  memcpy(myDense1_b, myModel_dense1_b, DENSE1_SIZE    * sizeof(float));
  memcpy(myDense2_w, myModel_dense2_w, DENSE2_WEIGHTS * sizeof(float));
  memcpy(myDense2_b, myModel_dense2_b, DENSE2_SIZE    * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, NUM_CLASSES    * sizeof(float));
  Serial.println("Baked-in weights loaded");
  myWeightsTrained = true;
  #if MY_NORM_FROM_DATA && defined(MY_HAS_BAKED_NORM)
  memcpy(myAccelMean, myModel_norm_mean, IMU_AXES * sizeof(float));      // v006
  memcpy(myAccelStd,  myModel_norm_std,  IMU_AXES * sizeof(float));
  Serial.println("Baked-in normalization loaded");
  #endif
#endif

#if MY_NORM_FROM_DATA
  myLoadCalibFromSD();                                 // v006: a saved normalization beats the baked-in one
#endif

  if (myLoadWeights()) {
    Serial.println("SD weights loaded - overriding baked-in weights");
  }

#if MY_USE_BLE
  myStartBle();                                        // v003
#endif

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);
  Serial.println("System ready - Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() {
  myPumpIncoming();             // v003: frames from the page (BLE queue + Web Serial lines)
  myHandleMenuNavigation();
  myLinkHeartbeat();            // v003: live ax,ay,az for the page while "debug frames" is on
}



// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: DATA COLLECTION FUNCTIONS                                       ██
// ██                                                                          ██
// ██  Captures one 1-second IMU window and saves it to SD as a .csv file.     ██
// ██  File format: one row per timestep, columns: ax,ay,az                    ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// Count .csv files in a class folder
int myCountSamples(int classIdx) {
  if (!mySDavailable) return 0;
  String path = "/motion/" + myClassLabels[classIdx];
  File root = SD.open(path);
  if (!root) return 0;
  int count = 0;
  while (File f = root.openNextFile()) {
    if (!f.isDirectory() && String(f.name()).endsWith(".csv")) count++;
    f.close();
  }
  root.close();
  return count;
}

// v003: first free file name sN.csv (v001 used the file COUNT, which could
// overwrite an existing file after a deletion).
bool myNewSamplePath(int classIdx, String* out) {
  if (!mySDavailable) return false;
  String folderPath = "/motion/" + myClassLabels[classIdx];
  if (!SD.exists("/motion")) SD.mkdir("/motion");
  if (!SD.exists(folderPath)) SD.mkdir(folderPath);
  for (int n = myCountSamples(classIdx); n < 100000; n++) {
    String p = folderPath + "/s" + String(n) + ".csv";
    if (!SD.exists(p)) { *out = p; return true; }
  }
  return false;
}

// v003: path of the n-th .csv in a class folder (same order as myCountSamples)
bool myNthSamplePath(int classIdx, int n, String* out) {
  if (!mySDavailable || classIdx < 0 || classIdx >= NUM_CLASSES || n < 0) return false;
  String path = "/motion/" + myClassLabels[classIdx];
  File root = SD.open(path);
  if (!root) return false;
  int count = 0;
  bool found = false;
  while (File f = root.openNextFile()) {
    String name = f.name();
    bool isCsv = !f.isDirectory() && name.endsWith(".csv");
    f.close();
    if (isCsv) {
      if (count == n) { *out = path + "/" + name; found = true; break; }
      count++;
    }
  }
  root.close();
  return found;
}

// v003: read one raw window (g) at SAMPLE_INTERVAL_MS spacing. Raw = not normalized.
void myCaptureWindow(float* w, bool echo) {
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    unsigned long tStart = millis();
    myReadAccel(w + t * IMU_AXES);
    // Brief echo every 10 samples
    if (echo && t % 10 == 0) Serial.printf("  t%02d: %.3f,%.3f,%.3f\n", t, w[t*IMU_AXES], w[t*IMU_AXES+1], w[t*IMU_AXES+2]);
    // Pace to SAMPLE_INTERVAL_MS
    long elapsed = millis() - tStart;
    if (elapsed < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - elapsed);
  }
}

// v003: one csv file, one row per timestep: ax,ay,az  (same format as v001)
bool myWriteSample(int classIdx, const float* w, String* pathOut) {
  String filePath;
  if (!myNewSamplePath(classIdx, &filePath)) return false;
  File f = SD.open(filePath, FILE_WRITE);
  if (!f) { Serial.println("ERROR: cannot open file for writing"); return false; }
  for (int t = 0; t < IMU_TIMESTEPS; t++)
    f.printf("%.5f,%.5f,%.5f\n", w[t*IMU_AXES], w[t*IMU_AXES+1], w[t*IMU_AXES+2]);
  f.close();
  Serial.printf("Saved: %s\n", filePath.c_str());
  if (pathOut) *pathOut = filePath;
  return true;
}

// Capture one IMU window: 40 samples at ~25 ms intervals, save to SD
bool myCaptureSample(int classIdx) {
  float w[INPUT_SIZE];
  Serial.printf("Capturing %d samples\n", IMU_TIMESTEPS);
  myCaptureWindow(w, true);
  return myWriteSample(classIdx, w, nullptr);
}

void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect samples");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("TAP (1-2 taps) = Capture 1-second window");
  Serial.println("LONG PRESS (3+ taps) = Exit to menu");
  Serial.println("Serial: 't'=capture, 'l'=exit");

  myResetTouchState();
  int captureCount = myCountSamples(classIdx);

  // Show OLED prompt
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 8,  myClassLabels[classIdx].c_str());
    u8g2.drawStr(0, 18, "TAP=Capture");
    u8g2.drawStr(0, 28, "HOLD=Exit");
    char buf[20]; snprintf(buf, sizeof(buf), "Count: %d", captureCount);
    u8g2.drawStr(0, 38, buf);
  } while (u8g2.nextPage());

  while (true) {
    myPumpIncoming();                                  // v003: keep the page link alive
    // Serial input
    if (myKeyAvailable()) {
      char c = myKeyRead();
      if (c == 'l' || c == 'L') { myResetMenuState(); return; }
      if (c == 't' || c == 'T') {
        Serial.println("Hold still... capturing in 1s");
        delay(1000);
        if (myCaptureSample(classIdx)) {
          captureCount++;
          Serial.printf("Total samples for %s: %d\n", myClassLabels[classIdx].c_str(), captureCount);
          u8g2.firstPage();
          do {
            u8g2.setFont(u8g2_font_5x7_tf);
            u8g2.drawStr(0, 8,  myClassLabels[classIdx].c_str());
            char buf[20]; snprintf(buf, sizeof(buf), "Saved: %d", captureCount);
            u8g2.drawStr(0, 20, buf);
            u8g2.drawStr(0, 32, "TAP=More");
          } while (u8g2.nextPage());
        }
      }
    }

    // Touch input
    int touchAction = myCheckTouchInput();
    if (touchAction == 2) { myResetMenuState(); return; }
    if (touchAction == 1) {
      Serial.println("Hold still... capturing in 1s");
      delay(1000);
      if (myCaptureSample(classIdx)) {
        captureCount++;
        Serial.printf("Total samples for %s: %d\n", myClassLabels[classIdx].c_str(), captureCount);
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.drawStr(0, 8,  myClassLabels[classIdx].c_str());
          char buf[20]; snprintf(buf, sizeof(buf), "Saved: %d", captureCount);
          u8g2.drawStr(0, 20, buf);
          u8g2.drawStr(0, 32, "TAP=More");
        } while (u8g2.nextPage());
      }
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: FORWARD PASS, BACKWARD PASS, TRAINING                          ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// Conv1D forward pass
// Input layout: [t0_ax, t0_ay, t0_az, t1_ax, ...]  (IMU_TIMESTEPS x IMU_AXES)
// Weight layout: [k x in_axis x out_filter]  index = (k*IMU_AXES + a)*CONV1_FILTERS + f
// Output layout: [step x filter]  index = step*CONV1_FILTERS + f
void myConv1DForward(float* input) {
  for (int s = 0; s < CONV1_OUT_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float sum = myConv1_b[f];
      for (int k = 0; k < CONV1_KERNEL; k++) {
        for (int a = 0; a < IMU_AXES; a++) {
          sum += input[(s + k) * IMU_AXES + a] * myConv1_w[(k * IMU_AXES + a) * CONV1_FILTERS + f];
        }
      }
      myConv1_output[s * CONV1_FILTERS + f] = myLeakyRelu(sum);
    }
  }
}

// Max-pool /2 along the time axis, per filter
void myPool1Forward() {
  for (int s = 0; s < POOL1_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float a = myConv1_output[(s * 2)     * CONV1_FILTERS + f];
      float b = myConv1_output[(s * 2 + 1) * CONV1_FILTERS + f];
      myPool1_output[s * CONV1_FILTERS + f] = max(a, b);
    }
  }
}

// Dense layer forward: output[j] = leaky_relu( sum_i(w[i*outSize+j] * input[i]) + b[j] )
void myDenseForward(float* input, int inSize,
                    float* w, float* b,
                    float* output, int outSize,
                    bool applyActivation) {
  for (int j = 0; j < outSize; j++) {
    float sum = b[j];
    for (int i = 0; i < inSize; i++) sum += input[i] * w[i * outSize + j];
    output[j] = applyActivation ? myLeakyRelu(sum) : sum;
  }
}

// Full forward pass: Conv1D -> Pool -> Dense1 -> Dense2 -> Output(softmax)
void myForwardPass(float* input) {
  myConv1DForward(input);
  myPool1Forward();
  myDenseForward(myPool1_output, CONV1_FLAT,  myDense1_w, myDense1_b, myDense1_output, DENSE1_SIZE, true);
  myDenseForward(myDense1_output, DENSE1_SIZE, myDense2_w, myDense2_b, myDense2_output, DENSE2_SIZE, true);
  myDenseForward(myDense2_output, DENSE2_SIZE, myOutput_w, myOutput_b, myFinal_output,  NUM_CLASSES, false);
  mySoftmax(myFinal_output, NUM_CLASSES);
}

// Cross-entropy loss for one sample (label is integer class index)
float myComputeLoss(int label) {
  float p = max(myFinal_output[label], 1e-7f);
  return -log(p);
}

// Adam update helper for one parameter array
void myAdamUpdate(float* w, float* grad, float* m, float* v, int size, float lr) {
  const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
  myAdamStep++;
  float bc1 = 1.0f - pow(beta1, myAdamStep);
  float bc2 = 1.0f - pow(beta2, myAdamStep);
  for (int i = 0; i < size; i++) {
    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
    float mHat = m[i] / bc1;
    float vHat = v[i] / bc2;
    w[i] -= lr * mHat / (sqrt(vHat) + eps);
  }
}

// Zero all gradient buffers
void myZeroGradients() {
  memset(myConv1_w_grad,  0, CONV1_WEIGHTS  * sizeof(float));
  memset(myConv1_b_grad,  0, CONV1_FILTERS  * sizeof(float));
  memset(myDense1_w_grad, 0, DENSE1_WEIGHTS * sizeof(float));
  memset(myDense1_b_grad, 0, DENSE1_SIZE    * sizeof(float));
  memset(myDense2_w_grad, 0, DENSE2_WEIGHTS * sizeof(float));
  memset(myDense2_b_grad, 0, DENSE2_SIZE    * sizeof(float));
  memset(myOutput_w_grad, 0, OUTPUT_WEIGHTS * sizeof(float));
  memset(myOutput_b_grad, 0, NUM_CLASSES    * sizeof(float));
}

// Backward pass for one sample, accumulates gradients into all layers
void myBackwardPass(float* input, int label) {
  // --- Output layer: softmax + cross-entropy combined ---
  for (int j = 0; j < NUM_CLASSES; j++)
    myOutput_delta[j] = myFinal_output[j] - (j == label ? 1.0f : 0.0f);

  for (int i = 0; i < DENSE2_SIZE; i++)
    for (int j = 0; j < NUM_CLASSES; j++)
      myOutput_w_grad[i * NUM_CLASSES + j] += myDense2_output[i] * myOutput_delta[j];
  for (int j = 0; j < NUM_CLASSES; j++) myOutput_b_grad[j] += myOutput_delta[j];

  // --- Dense2 ---
  for (int i = 0; i < DENSE2_SIZE; i++) {
    float sum = 0;
    for (int j = 0; j < NUM_CLASSES; j++) sum += myOutput_w[i * NUM_CLASSES + j] * myOutput_delta[j];
    myDense2_delta[i] = sum * myLeakyReluDeriv(myDense2_output[i]);
  }
  for (int i = 0; i < DENSE1_SIZE; i++)
    for (int j = 0; j < DENSE2_SIZE; j++)
      myDense2_w_grad[i * DENSE2_SIZE + j] += myDense1_output[i] * myDense2_delta[j];
  for (int j = 0; j < DENSE2_SIZE; j++) myDense2_b_grad[j] += myDense2_delta[j];

  // --- Dense1 ---
  for (int i = 0; i < DENSE1_SIZE; i++) {
    float sum = 0;
    for (int j = 0; j < DENSE2_SIZE; j++) sum += myDense2_w[i * DENSE2_SIZE + j] * myDense2_delta[j];
    myDense1_delta[i] = sum * myLeakyReluDeriv(myDense1_output[i]);
  }
  for (int i = 0; i < CONV1_FLAT; i++)
    for (int j = 0; j < DENSE1_SIZE; j++)
      myDense1_w_grad[i * DENSE1_SIZE + j] += myPool1_output[i] * myDense1_delta[j];
  for (int j = 0; j < DENSE1_SIZE; j++) myDense1_b_grad[j] += myDense1_delta[j];

  // --- Max-pool backward: route gradient to whichever input was the max ---
  for (int s = 0; s < POOL1_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float grad = 0;
      for (int j = 0; j < DENSE1_SIZE; j++)
        grad += myDense1_w[((s * CONV1_FILTERS + f)) * DENSE1_SIZE + j] * myDense1_delta[j];
      myPool1_delta[s * CONV1_FILTERS + f] = grad;

      float a = myConv1_output[(s * 2)     * CONV1_FILTERS + f];
      float b = myConv1_output[(s * 2 + 1) * CONV1_FILTERS + f];
      myConv1_delta[(s * 2)     * CONV1_FILTERS + f] = (a >= b) ? grad : 0.0f;
      myConv1_delta[(s * 2 + 1) * CONV1_FILTERS + f] = (b >  a) ? grad : 0.0f;
    }
  }

  // --- Conv1D backward ---
  for (int s = 0; s < CONV1_OUT_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float delta = myConv1_delta[s * CONV1_FILTERS + f]
                    * myLeakyReluDeriv(myConv1_output[s * CONV1_FILTERS + f]);
      myConv1_b_grad[f] += delta;
      for (int k = 0; k < CONV1_KERNEL; k++) {
        for (int a = 0; a < IMU_AXES; a++) {
          myConv1_w_grad[(k * IMU_AXES + a) * CONV1_FILTERS + f] +=
            input[(s + k) * IMU_AXES + a] * delta;
        }
      }
    }
  }
}

// v003: raw csv -> buf (INPUT_SIZE floats, NOT normalized). The page's GET uses this.
bool myReadSampleCsv(const char* path, float* buf) {
  File f = SD.open(path);
  if (!f) return false;
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    for (int a = 0; a < IMU_AXES; a++) {
      buf[t * IMU_AXES + a] = f.parseFloat();
      if (a < IMU_AXES - 1) {
        // consume the comma
        while (f.available() && f.peek() == ',') f.read();
      }
    }
    // consume newline
    while (f.available() && (f.peek() == '\n' || f.peek() == '\r')) f.read();
  }
  f.close();
  return true;
}

// Load one .csv sample from SD into buf (INPUT_SIZE floats) then normalize
bool myLoadSampleFromFile(const char* path, float* buf) {
  if (!myReadSampleCsv(path, buf)) return false;
  myNormalizeInput(buf);
  return true;
}

// v003: the training body, shared by the menu (myActionTrain) and the page ("TRAIN").
// Returns true if it trained. Sends "EP ..." lines to the page after each epoch.
bool myTrainCore() {
  if (!mySDavailable) {
    Serial.println("No SD - cannot train");
    myReply("ERR no SD card - device training needs the SD samples");
    return false;
  }

  // Build training list
  myTrainingData.clear();
  int classCounts[NUM_CLASSES] = {};
  for (int c = 0; c < NUM_CLASSES; c++) {
    String path = "/motion/" + myClassLabels[c];
    File root = SD.open(path);
    if (!root) continue;
    while (File file = root.openNextFile()) {
      String name = file.name();
      if (!file.isDirectory() && name.endsWith(".csv")) {
        myTrainingData.push_back({path + "/" + name, c});
        classCounts[c]++;
      }
      file.close();
    }
    root.close();
  }

  Serial.println("\n=== Training ===");
  for (int c = 0; c < NUM_CLASSES; c++)
    Serial.printf("  %s: %d samples\n", myClassLabels[c].c_str(), classCounts[c]);
  if (myTrainingData.empty()) {
    myReply("ERR no samples on the SD card");
    return false;
  }

  // Shuffle and split validation
  std::random_shuffle(myTrainingData.begin(), myTrainingData.end());
  int valCount = 0;
  std::vector<TrainingItem> myValData;
  if (VALIDATION_SAMPLES > 0) {
    // Hold out last VALIDATION_SAMPLES per class
    int heldOut[NUM_CLASSES] = {};
    std::vector<TrainingItem> trainOnly;
    for (auto& item : myTrainingData) {
      if (heldOut[item.label] < VALIDATION_SAMPLES) {
        myValData.push_back(item);
        heldOut[item.label]++;
        valCount++;
      } else {
        trainOnly.push_back(item);
      }
    }
    myTrainingData = trainOnly;
  }
  Serial.printf("Training: %d samples  Validation: %d samples\n",
                (int)myTrainingData.size(), valCount);

  // Training loop
  float* myBatchBuf = (float*)ps_malloc(INPUT_SIZE * sizeof(float));
  if (!myBatchBuf) { Serial.println("malloc failed"); myReply("ERR malloc"); return false; }

#if MY_NORM_FROM_DATA
  myComputeNormFromData(myTrainingData, myBatchBuf);   // v006: BEFORE the first sample is normalized
#endif
#if MY_TRAIN_FRESH
  myInitWeights();                                     // v006: new run, new random weights, empty Adam history
  myResetAdam();
#endif

  myStopRequested = false;
  int epochsDone = 0;
  for (int epoch = 0; epoch < TARGET_EPOCHS; epoch++) {
    std::random_shuffle(myTrainingData.begin(), myTrainingData.end());
    float epochLoss = 0;
    int   correct   = 0;
    int   processed = 0;
    myZeroGradients();

    for (int si = 0; si < (int)myTrainingData.size(); si++) {
      myCheckTouchBackground();  // keep touch responsive
      myPumpIncoming();                                              // v003
      if (myKeyAvailable() && myKeyRead() == 'x') myStopRequested = true;   // v003
      if (myStopRequested) break;                                    // v003
      if (!myLoadSampleFromFile(myTrainingData[si].path.c_str(), myBatchBuf)) continue;

      myForwardPass(myBatchBuf);
      int label = myTrainingData[si].label;
      epochLoss += myComputeLoss(label);

      // Argmax for accuracy
      int pred = 0;
      for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;
      if (pred == label) correct++;

      myBackwardPass(myBatchBuf, label);
      processed++;

      // Apply gradients at end of each mini-batch
      if ((si + 1) % BATCH_SIZE == 0 || si == (int)myTrainingData.size() - 1) {
        float scale = 1.0f / processed;
        for (int k = 0; k < CONV1_WEIGHTS;  k++) myConv1_w_grad[k]  *= scale;
        for (int k = 0; k < CONV1_FILTERS;  k++) myConv1_b_grad[k]  *= scale;
        for (int k = 0; k < DENSE1_WEIGHTS; k++) myDense1_w_grad[k] *= scale;
        for (int k = 0; k < DENSE1_SIZE;    k++) myDense1_b_grad[k] *= scale;
        for (int k = 0; k < DENSE2_WEIGHTS; k++) myDense2_w_grad[k] *= scale;
        for (int k = 0; k < DENSE2_SIZE;    k++) myDense2_b_grad[k] *= scale;
        for (int k = 0; k < OUTPUT_WEIGHTS; k++) myOutput_w_grad[k] *= scale;
        for (int k = 0; k < NUM_CLASSES;    k++) myOutput_b_grad[k] *= scale;

        myAdamUpdate(myConv1_w,  myConv1_w_grad,  myConv1_w_m,  myConv1_w_v,  CONV1_WEIGHTS,  LEARNING_RATE);
        myAdamUpdate(myConv1_b,  myConv1_b_grad,  myConv1_b_m,  myConv1_b_v,  CONV1_FILTERS,  LEARNING_RATE);
        myAdamUpdate(myDense1_w, myDense1_w_grad, myDense1_w_m, myDense1_w_v, DENSE1_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myDense1_b, myDense1_b_grad, myDense1_b_m, myDense1_b_v, DENSE1_SIZE,    LEARNING_RATE);
        myAdamUpdate(myDense2_w, myDense2_w_grad, myDense2_w_m, myDense2_w_v, DENSE2_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myDense2_b, myDense2_b_grad, myDense2_b_m, myDense2_b_v, DENSE2_SIZE,    LEARNING_RATE);
        myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, OUTPUT_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES,    LEARNING_RATE);

        myZeroGradients();
        processed = 0;
      }
    }
    if (myStopRequested) { Serial.println("Training stopped"); break; }   // v003

    // Validation
    float valAcc = 0;
    if (valCount > 0) {
      int valCorrect = 0;
      for (auto& vi : myValData) {
        if (!myLoadSampleFromFile(vi.path.c_str(), myBatchBuf)) continue;
        myForwardPass(myBatchBuf);
        int pred = 0;
        for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;
        if (pred == vi.label) valCorrect++;
      }
      valAcc = 100.0f * valCorrect / valCount;
    }

    float trainAcc = 100.0f * correct / max((int)myTrainingData.size(), 1);
    float avgLoss  = epochLoss / max((int)myTrainingData.size(), 1);
    Serial.printf("Epoch %2d/%d  Loss=%.4f  TrainAcc=%.1f%%  ValAcc=%.1f%%\n",
                  epoch + 1, TARGET_EPOCHS, avgLoss, trainAcc, valAcc);
    myReply("EP %d %d %.4f %.1f %.1f", epoch + 1, TARGET_EPOCHS, avgLoss, trainAcc, valAcc);   // v003
    epochsDone++;

    // OLED progress
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      char buf[24];
      snprintf(buf, sizeof(buf), "Ep %d/%d", epoch + 1, TARGET_EPOCHS);
      u8g2.drawStr(0, 8, buf);
      snprintf(buf, sizeof(buf), "Tr %.0f%%", trainAcc);
      u8g2.drawStr(0, 18, buf);
      if (valCount > 0) { snprintf(buf, sizeof(buf), "Val %.0f%%", valAcc); u8g2.drawStr(0, 28, buf); }
    } while (u8g2.nextPage());
  }

  free(myBatchBuf);
  myStopRequested = false;
  myWeightsTrained = true;
  mySaveWeights();
  mySaveCalib();                                       // v006: the normalization these weights were trained with
  myReplyCal();                                        // v006: tell the page (CAL line)
  Serial.println("Training complete. Weights saved.");
  myReply("DONE train %d epochs, weights saved=%d", epochsDone, mySDavailable ? 1 : 0);   // v003
  return true;
}

void myActionTrain() {
  myTrainCore();                                                     // v003: body moved to myTrainCore()
  u8g2.firstPage();
  do { u8g2.setFont(u8g2_font_5x7_tf); u8g2.drawStr(0, 15, "Training done!"); u8g2.drawStr(0, 28, "Weights saved"); } while (u8g2.nextPage());
  delay(2000);
  myResetMenuState();
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE FUNCTIONS                                             ██
// ██                                                                          ██
// ██  Continuously captures 1-second IMU windows and classifies them.         ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


void myActionInfer() {
  if (!myWeightsTrained) {
    Serial.println("No trained weights - train or load first");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No weights!"); u8g2.drawStr(0, 28, "Train first"); } while (u8g2.nextPage());
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Inference mode (tap A0 or 'l' to exit)");
  Serial.println("Majority vote over 3 consecutive windows.");

  float myLiveBuf[INPUT_SIZE];
  int   windowCount  = 0;
  int   voteBuf[3]   = {0, 0, 0};   // rolling window of last 3 predictions
  int   voteIdx      = 0;
  int   finalPred    = 0;

  while (true) {
    myPumpIncoming();                                  // v003
    // Check exit
    if (myCheckTouchInput() == 2) { myResetMenuState(); return; }
    if (myKeyAvailable()) { char c = myKeyRead(); if (c == 'l' || c == 'L') { myResetMenuState(); return; } }

    // Capture one 1-second window
    myCaptureWindow(myLiveBuf, false);                 // v003

    myNormalizeInput(myLiveBuf);
    myForwardPass(myLiveBuf);

    // Argmax for this window
    int rawPred = 0;
    for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[rawPred]) rawPred = j;
    windowCount++;

    // Store in rolling vote buffer
    voteBuf[voteIdx % 3] = rawPred;
    voteIdx++;

    // Majority vote over last 3 windows
    int votes[NUM_CLASSES] = {};
    for (int v = 0; v < 3; v++) votes[voteBuf[v]]++;
    finalPred = 0;
    for (int j = 1; j < NUM_CLASSES; j++) if (votes[j] > votes[finalPred]) finalPred = j;

    Serial.printf("Win %d raw=%s | vote=%s | All:",
                  windowCount, myClassLabels[rawPred].c_str(), myClassLabels[finalPred].c_str());
    for (int j = 0; j < NUM_CLASSES; j++) Serial.printf(" %.0f%%", myFinal_output[j] * 100);
    Serial.println();

    // OLED: show voted result and raw confidence
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.drawStr(0, 8,  "Motion:");
      u8g2.drawStr(0, 18, myClassLabels[finalPred].c_str());
      char buf[20];
      snprintf(buf, sizeof(buf), "raw %.0f%% #%d", myFinal_output[rawPred] * 100, windowCount);
      u8g2.drawStr(0, 28, buf);
    } while (u8g2.nextPage());
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: MENU SYSTEM FUNCTIONS                                           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


void myResetMenuState() {
  myIsSelected = false;
  myBusy = false;                                      // v003
  myResetTouchState();
  myLastActivityTime = millis();
  myDrawMenu();
}

void myDrawMenu() {
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    String label =
      (i <= NUM_CLASSES) ? myClassLabels[i - 1] :
      (i == NUM_CLASSES + 1) ? "Train" : "Infer";
    Serial.printf("%s%d. %s\n", (i == myMenuIndex) ? " > " : "   ", i, label.c_str());
  }
  Serial.println("Commands: t=next  l=select");

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
    int myStartItem = (myMenuIndex <= NUM_CLASSES) ? 1 : myMenuIndex - 2;
    for (int i = 0; i < 3; i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;
      String label =
        (cur <= NUM_CLASSES) ? myClassLabels[cur - 1] :
        (cur == NUM_CLASSES + 1) ? "Train" : "Infer";
      int y = 18 + i * 9;
      u8g2.drawStr(0, y, ((cur == myMenuIndex) ? "> " + label : "  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

void myExecuteMenuItem(int idx) {
  myBusy = true;                                       // v003: page commands answer "BUSY" meanwhile
  if      (idx <= NUM_CLASSES)        myActionCollect(idx - 1);
  else if (idx == NUM_CLASSES + 1)    myActionTrain();
  else                                myActionInfer();
  myBusy = false;
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis = millis();

  if (!myIsSelected && myKeyAvailable()) {             // v003: was Serial.available()
    char c = myKeyRead();
    if (c >= '1' && c <= '9') {
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) {
        myMenuIndex = newIndex;
        myIsSelected = true;
        myLastActivityTime = myCurrentMillis;
        myExecuteMenuItem(myMenuIndex);
      }
    }
    else if (c == 't' || c == 'T') {
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (c == 'l' || c == 'L') {
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }

  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();
    if (touchAction == 1) {
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (touchAction == 2) {
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }
}



// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 5 (v003): LINK TO THE WEB PAGE  (WebBLE + Web Serial)              ██
// ██                                                                          ██
// ██  Same frames on both transports. Frames from BLE arrive in a callback    ██
// ██  and are only QUEUED there; all work happens in loop() / myPumpIncoming. ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// ==LINK START==
// Standard CRC-32 (the same one zip and the page use)
uint32_t myCrc32(const uint8_t* d, size_t n) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; i++) {
    c ^= d[i];
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
  }
  return ~c;
}

void myPut32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)((v >> 8) & 0xFF);
  p[2] = (uint8_t)((v >> 16) & 0xFF); p[3] = (uint8_t)((v >> 24) & 0xFF);
}

uint32_t myGet32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

#if MY_USE_BLE
// v006: one notification, retried for up to a second while the BLE stack's buffers are full.
bool mySendNotify(const uint8_t* p, size_t n) {
  if (!myBleConnected || !myEvtChar) return false;
  myEvtChar->setValue(p, n);
  unsigned long t0 = millis();
  bool ok = myEvtChar->notify();
  while (!ok && myBleConnected && millis() - t0 < 1000) { delay(2); ok = myEvtChar->notify(); }
  return ok;
}
#endif

// Send one frame on the transport the last command came in on.
bool mySendFrame(const uint8_t* d, size_t n) {
  if (myReplyVia == 1) {
#if MY_USE_BLE
    if (!myBleConnected || !myEvtChar) return false;
    // v006: a notification carries at most (MTU - 3) bytes. Every packet is [kind][bytes]:
    //   kind 0 = a whole frame, 1 = first piece, 2 = middle piece, 3 = last piece.
    // The page puts the pieces back together. (v005 sent the whole frame and the phone cut it off.)
    size_t pl = (size_t)myBleMtu;
    pl = (pl > 3) ? pl - 3 : 20;
    if (pl > MY_FRAME_MAX + 1) pl = MY_FRAME_MAX + 1;
    if (pl < 8) pl = 20;
    size_t room = pl - 1;
    uint8_t pkt[MY_FRAME_MAX + 1];
    if (n <= room) {
      pkt[0] = 0;
      memcpy(pkt + 1, d, n);
      return mySendNotify(pkt, n + 1);
    }
    size_t off = 0;
    while (off < n) {
      size_t len = n - off;
      if (len > room) len = room;
      pkt[0] = (off == 0) ? 1 : ((off + len >= n) ? 3 : 2);
      memcpy(pkt + 1, d + off, len);
      if (!mySendNotify(pkt, len + 1)) return false;   // a piece was lost: the page drops the half frame, the blob protocol resends
      off += len;
    }
    return true;
#else
    return false;
#endif
  }
  if (myReplyVia == 2) {
    char line[MY_FRAME_MAX * 2 + 16];
    size_t ol = 0;
    memcpy(line, "@B ", 3);
    if (mbedtls_base64_encode((unsigned char*)line + 3, sizeof(line) - 5, &ol, d, n) != 0) return false;
    line[3 + ol] = '\n';
    Serial.write((const uint8_t*)line, 3 + ol + 1);   // one write so other prints do not split the line
    return true;
  }
  return false;
}

// Text reply line, e.g. myReply("OK model saved=%d", 1)
void myReply(const char* fmt, ...) {
  uint8_t f[MY_FRAME_MAX];
  f[0] = MY_F_RESP;
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf((char*)f + 1, MY_FRAME_MAX - 1, fmt, ap);
  va_end(ap);
  if (n < 0) return;
  if (n > MY_FRAME_MAX - 2) n = MY_FRAME_MAX - 2;
  mySendFrame(f, 1 + n);
}

// next = the next chunk number I expect; status 0 ok, 1 crc error, 2 abort
void myAck(uint16_t next, uint8_t status) {
  uint8_t f[4] = { MY_F_ACK, (uint8_t)(next & 0xFF), (uint8_t)(next >> 8), status };
  mySendFrame(f, 4);
}

// Send a blob (kind 'M' model, 'S' sample window) with acks every MY_WIN chunks.
// Returns false if the page stopped answering or refused it.
bool mySendBlob(uint8_t kind, uint8_t id, const uint8_t* data, uint32_t total) {
  mySending = true;
  uint32_t crc = myCrc32(data, total);
  uint16_t nCh = (uint16_t)((total + MY_CHUNK_DATA - 1) / MY_CHUNK_DATA);
  uint8_t f[MY_FRAME_MAX];
  f[0] = MY_F_HEAD; f[1] = kind; f[2] = id;
  myPut32(f + 3, total); myPut32(f + 7, crc);
  mySendFrame(f, 11);

  uint16_t acked = 0;
  int tries = 0;
  while (acked < nCh) {
    uint16_t end = (uint16_t)(acked + MY_WIN);
    if (end > nCh) end = nCh;
    myTxAckNext = 0xFFFF;
    myTxAckStatus = 0;
    for (uint16_t s = acked; s < end; s++) {
      uint32_t off = (uint32_t)s * MY_CHUNK_DATA;
      uint32_t n = total - off;
      if (n > MY_CHUNK_DATA) n = MY_CHUNK_DATA;
      f[0] = MY_F_DATA; f[1] = (uint8_t)(s & 0xFF); f[2] = (uint8_t)(s >> 8);
      memcpy(f + 3, data + off, n);
      mySendFrame(f, 3 + n);
      delay(8);
    }
    unsigned long t0 = millis();
    while (myTxAckNext == 0xFFFF && millis() - t0 < MY_ACK_TIMEOUT_MS) { myPumpIncoming(); delay(2); }
    if (myTxAckNext == 0xFFFF) {                       // no answer: resend this window
      if (++tries > 10) { mySending = false; return false; }
      if (acked == 0) {                                // the page may have missed the header
        f[0] = MY_F_HEAD; f[1] = kind; f[2] = id; myPut32(f + 3, total); myPut32(f + 7, crc);
        mySendFrame(f, 11);
      }
      continue;
    }
    if (myTxAckStatus == 2) { mySending = false; return false; }   // the page aborted
    if (myTxAckNext > acked) { acked = myTxAckNext; tries = 0; }
    else {                                             // "still expecting chunk N": something was lost
      if (++tries > 10) { mySending = false; return false; }
      unsigned long t1 = millis();                     // the receiver sends one such nack per later chunk:
      while (millis() - t1 < 80) { myPumpIncoming(); delay(2); }   // let this round's duplicates drain, then resend
    }
  }
  mySending = false;
  return true;
}

// ---- receiving a blob from the page ----
void myOnHead(const uint8_t* d) {
  uint8_t  kind  = d[1];
  uint8_t  id    = d[2];
  uint32_t total = myGet32(d + 3);
  uint32_t crc   = myGet32(d + 7);
  if (myBusy) { myReply("ERR busy - try again when the device is idle"); myAck(0, 2); return; }
  bool ok = false;
  if (kind == 'W')      ok = (total == MY_PACKAGE_BYTES);
  else if (kind == 'S') ok = (total == INPUT_SIZE * 4 && id < NUM_CLASSES);
  else if (kind == 'C') ok = (total > 0 && total <= 4096);
  if (!ok) {
    myReply("ERR refused blob %c id=%u size=%lu - this sketch needs W=%u S=%u (classes %u)",
            (char)kind, (unsigned)id, (unsigned long)total,
            (unsigned)MY_PACKAGE_BYTES, (unsigned)(INPUT_SIZE * 4), (unsigned)NUM_CLASSES);
    myAck(0, 2);
    return;
  }
  myRxKind = kind; myRxId = id; myRxTotal = total; myRxCrc = crc;
  myRxGot = 0; myRxNext = 0; myRxLastMs = millis();
  myRxActive = true;
  myAck(0, 0);
}

void myOnData(const uint8_t* d, size_t n) {
  if (!myRxActive) return;
  uint16_t seq = (uint16_t)(d[1] | (d[2] << 8));
  myRxLastMs = millis();
  if (seq != myRxNext) { myAck(myRxNext, 0); return; }      // out of order: say what I expect
  uint32_t off = (uint32_t)seq * MY_CHUNK_DATA;
  uint32_t len = (uint32_t)(n - 3);
  if (off + len > myRxTotal) { myRxActive = false; myAck(myRxNext, 2); return; }
  memcpy(myBlobBuf + off, d + 3, len);
  myRxNext++;
  myRxGot += len;
  if (myRxGot >= myRxTotal) {
    myRxActive = false;
    if (myCrc32(myBlobBuf, myRxTotal) == myRxCrc) {
      myAck(myRxNext, 0);
      myDispatchBlob(myRxKind, myRxId, myRxTotal);
    } else {
      myAck(myRxNext, 1);
      myReply("ERR crc mismatch - transfer discarded");
    }
  } else if ((myRxNext % MY_WIN) == 0) {
    myAck(myRxNext, 0);
  }
}

void myHandleFrame(const uint8_t* d, size_t n) {
  if (n < 1) return;
  switch (d[0]) {
    case MY_F_HEAD: if (n == 11) myOnHead(d); break;
    case MY_F_DATA: if (n > 3)   myOnData(d, n); break;
    case MY_F_ACK:  if (n >= 4) {                      // keep the HIGHEST ack of this round: a late duplicate nack must not undo progress
      uint16_t nx = (uint16_t)(d[1] | (d[2] << 8));
      if (d[3] == 2) { myTxAckStatus = 2; if (myTxAckNext == 0xFFFF) myTxAckNext = nx; }
      else if (myTxAckNext == 0xFFFF || nx >= myTxAckNext) { if (myTxAckStatus != 2 || myTxAckNext == 0xFFFF) myTxAckStatus = d[3]; myTxAckNext = nx; }
    } break;
    case MY_F_TEXT: {
      char s[MY_FRAME_MAX];
      memcpy(s, d + 1, n - 1);
      s[n - 1] = 0;
      myHandleCommand(s);
    } break;
    default: break;
  }
}

// BLE callback -> queue. Called from the BLE task, so it only copies.
void myQPush(const uint8_t* d, size_t n, uint8_t via) {
  if (n == 0 || n > MY_FRAME_MAX) return;
  uint8_t next = (uint8_t)((myQHead + 1) % MY_QN);
  if (next == myQTail) return;                         // full: drop it, the ack protocol resends
  myQ[myQHead].len = (uint8_t)n;
  myQ[myQHead].via = via;
  memcpy(myQ[myQHead].d, d, n);
  myQHead = next;
}

// Web Serial: lines that start with '@' are frames. Anything else is left for the menu.
void myPollSerialFrames() {
  while (Serial.available()) {
    if (mySerLen == 0 && Serial.peek() != '@') return;
    int c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (mySerLen > 0) {
        mySerLine[mySerLen] = 0;
        uint8_t f[MY_FRAME_MAX + 4];
        size_t ol = 0;
        int len = mySerLen;
        mySerLen = 0;                                  // free the line buffer before handling
        if (len > 3 && mySerLine[1] == 'B' && mySerLine[2] == ' ' &&
            mbedtls_base64_decode(f, sizeof(f), &ol, (const unsigned char*)mySerLine + 3, len - 3) == 0 && ol > 0) {
          myReplyVia = 2;
          myHandleFrame(f, ol);
        }
        if (mySending && myTxAckNext != 0xFFFF) return;   // the awaited ack is in: leave the rest for after this command
      }
      continue;
    }
    if (mySerLen < (int)sizeof(mySerLine) - 1) mySerLine[mySerLen++] = (char)c;
    else mySerLen = 0;                                 // too long: drop the line
  }
}

// Handle everything waiting. Safe to call from long loops (training, sending).
void myPumpIncoming() {
  myPollSerialFrames();
  while (myQTail != myQHead) {
    MyFrame fr = myQ[myQTail];                         // copy first, then advance (handlers may re-enter)
    myQTail = (uint8_t)((myQTail + 1) % MY_QN);
    myReplyVia = fr.via;
    myHandleFrame(fr.d, fr.len);
    if (mySending && myTxAckNext != 0xFFFF) break;     // same rule as the serial path
  }
  if (myRxActive && millis() - myRxLastMs > 6000) {
    myRxActive = false;
    myReply("ERR receive timeout - transfer dropped");
  }
}
// ==LINK END==


// ---- BLE server (NimBLE-Arduino 2.x calls, as used in the fusion firmware) ----
#if MY_USE_BLE
class MyBleServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* s, NimBLEConnInfo& info) override {
    myBleConnected = true;
    myBleMtu = 23;                                     // v006: until the phone negotiates more
    s->updateConnParams(info.getConnHandle(), 12, 24, 0, 400);   // v006: ask for a 15-30 ms interval (faster transfers)
    Serial.println("Browser connected (BLE)");
  }
  // v006: no "override" on purpose: if your NimBLE version has no such callback this is simply never called
  // and the sketch keeps the safe 20 byte notifications.
  void onMTUChange(uint16_t MTU, NimBLEConnInfo& info) {
    myBleMtu = MTU;
    Serial.printf("BLE MTU is now %u (notifications carry %u bytes)\n", (unsigned)MTU, (unsigned)(MTU > 3 ? MTU - 3 : 20));
  }
  void onDisconnect(NimBLEServer* s, NimBLEConnInfo& info, int reason) override {
    myBleConnected = false;
    myRxActive = false;
    Serial.println("Browser disconnected - advertising again");
    NimBLEDevice::startAdvertising();
  }
};

class MyBleCmdCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& info) override {
    std::string v = c->getValue();
    myQPush((const uint8_t*)v.data(), v.size(), 1);    // work happens in loop()
  }
};

void myStartBle() {
  NimBLEDevice::init(MY_DEVICE_NAME);
  NimBLEDevice::setMTU(247);
  myBleServer = NimBLEDevice::createServer();
  myBleServer->setCallbacks(new MyBleServerCallbacks());
  NimBLEService* svc = myBleServer->createService(MY_SVC_UUID);
  myCmdChar = svc->createCharacteristic(MY_CMD_UUID, NIMBLE_PROPERTY::WRITE);
  myEvtChar = svc->createCharacteristic(MY_EVT_UUID, NIMBLE_PROPERTY::NOTIFY);
  myCmdChar->setCallbacks(new MyBleCmdCallbacks());
  svc->start();

  // Advertise the NAME only: a 128-bit UUID + name overflows the 31-byte packet and
  // advertising then fails silently (the page filters by name prefix anyway).
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->setName(MY_DEVICE_NAME);
  if (adv->start()) {
    Serial.print("BLE advertising as \"");
    Serial.print(MY_DEVICE_NAME);
    Serial.println("\" - open index-v006.html and press Connect BLE");
  } else {
    Serial.println("ERROR: BLE advertising failed to start - the page will not see this board");
  }
}
#endif


// ---- what the page can ask for ----
// Text commands:  STATUS  DBG 1|0  STOP  CAPTURE c  LIST (inside STATUS)  GET c n  DEL c n
//                 TRAIN  INFER  GETMODEL  CALIB
// Blobs from the page: 'W' model package, 'S' sample window for class id, 'C' config.json text.
// v005: CRC-32 of the model package (weights in file order, then mean, then std),
// the SAME bytes as myPackToBuf() but WITHOUT copying them, so it is safe to call
// at any time (the blob buffer may be busy with a transfer).
static uint32_t myCrcFeed(uint32_t c, const uint8_t* d, size_t n) {
  for (size_t i = 0; i < n; i++) {
    c ^= d[i];
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
  }
  return c;
}
uint32_t myModelCrc() {
  const float* src[8] = { myConv1_w, myConv1_b, myDense1_w, myDense1_b, myDense2_w, myDense2_b, myOutput_w, myOutput_b };
  const int    cnt[8] = { CONV1_WEIGHTS, CONV1_FILTERS, DENSE1_WEIGHTS, DENSE1_SIZE, DENSE2_WEIGHTS, DENSE2_SIZE, OUTPUT_WEIGHTS, NUM_CLASSES };
  uint32_t c = 0xFFFFFFFFu;
  for (int b = 0; b < 8; b++) c = myCrcFeed(c, (const uint8_t*)src[b], cnt[b] * 4);
  c = myCrcFeed(c, (const uint8_t*)myAccelMean, IMU_AXES * 4);
  c = myCrcFeed(c, (const uint8_t*)myAccelStd,  IMU_AXES * 4);
  return ~c;
}

void myReplyInfo() {
#if MY_USE_BLE
  int myMtuNow = (myReplyVia == 1) ? (int)myBleMtu : 0;                                   // v006
#else
  int myMtuNow = 0;
#endif
  myReply("INFO T=%d A=%d K=%d F=%d D1=%d D2=%d C=%d INT=%d W=%d SD=%d TR=%d MC=%lu MTU=%d",   // v005: MC=, v006: MTU=
          IMU_TIMESTEPS, IMU_AXES, CONV1_KERNEL, CONV1_FILTERS, DENSE1_SIZE, DENSE2_SIZE,
          NUM_CLASSES, SAMPLE_INTERVAL_MS, MY_WEIGHT_FLOATS, mySDavailable ? 1 : 0, myWeightsTrained ? 1 : 0,
          (unsigned long)myModelCrc(), myMtuNow);
  myReply("HELLO %s firmware-v006", MY_DEVICE_NAME);                                      // v006
  char buf[MY_FRAME_MAX];
  int n = snprintf(buf, sizeof(buf), "LBL ");
  for (int i = 0; i < NUM_CLASSES && n < (int)sizeof(buf) - 2; i++)
    n += snprintf(buf + n, sizeof(buf) - n, "%s%s", i ? "," : "", myClassLabels[i].c_str());
  myReply("%s", buf);
  n = snprintf(buf, sizeof(buf), "LST");
  for (int i = 0; i < NUM_CLASSES && n < (int)sizeof(buf) - 8; i++)
    n += snprintf(buf + n, sizeof(buf) - n, " %d", myCountSamples(i));
  myReply("%s", buf);
  myReplyCal();                                        // last line: the page waits for it
}

void myReplyCal() {
  myReply("CAL %.5f %.5f %.5f %.5f %.5f %.5f",
          myAccelMean[0], myAccelMean[1], myAccelMean[2], myAccelStd[0], myAccelStd[1], myAccelStd[2]);
}

void myImportPackage() {
  if (!myPackFromBuf((const float*)myBlobBuf)) { myReply("ERR model holds NaN or Infinity - rejected"); return; }
  myWeightsTrained = true;
  mySaveWeights();
  mySaveCalib();
  myReply("OK model loaded, saved to SD=%d", mySDavailable ? 1 : 0);
}

void myStoreSample(int classIdx) {
  if (!mySDavailable) { myReply("ERR no SD card - sample not stored"); return; }
  String p;
  if (myWriteSample(classIdx, (const float*)myBlobBuf, &p)) myReply("OK put %d %s", classIdx, p.c_str());
  else myReply("ERR could not write the sample");
}

void myStoreConfig(uint32_t total) {
  myBlobBuf[total] = 0;
  if (mySDavailable) {
    if (!SD.exists("/header")) SD.mkdir("/header");
    File f = SD.open("/header/config.json", FILE_WRITE);
    if (f) { f.write(myBlobBuf, total); f.close(); }
  }
  myApplyConfig((const char*)myBlobBuf);
  myReply("OK config %s", mySDavailable ? "saved" : "applied (no SD)");
}

void myDispatchBlob(uint8_t kind, uint8_t id, uint32_t total) {
  if (kind == 'W')      myImportPackage();
  else if (kind == 'S') myStoreSample(id);
  else if (kind == 'C') myStoreConfig(total);
}

void myHandleCommand(char* s) {
  if (!strncmp(s, "DBG ", 4)) { myLinkDebugOn = (s[4] == '1'); myDebugLastMs = millis(); return; }
  if (!strcmp(s, "STOP"))     { myStopRequested = true; return; }
  if (!strcmp(s, "STATUS"))   { myReplyInfo(); return; }
  if (myBusy || myRxActive)   { myReply("BUSY"); return; }
  myBusy = true;
  if (!strncmp(s, "CAPTURE ", 8)) {
    int c = atoi(s + 8);
    if (c < 0 || c >= NUM_CLASSES) {
      myReply("ERR class %d out of range", c);
    } else {
      float w[INPUT_SIZE];
      myCaptureWindow(w, false);
      String p;
      if (mySDavailable && myWriteSample(c, w, &p)) myReply("OK cap %d %s", c, p.c_str());
      else myReply("WARN cap %d not saved (no SD or write failed)", c);
      mySendBlob('S', (uint8_t)c, (const uint8_t*)w, INPUT_SIZE * 4);
    }
  } else if (!strncmp(s, "GET ", 4)) {
    int c = -1, n = -1;
    sscanf(s + 4, "%d %d", &c, &n);
    String p;
    if (myNthSamplePath(c, n, &p) && myReadSampleCsv(p.c_str(), (float*)myBlobBuf))
      mySendBlob('S', (uint8_t)c, myBlobBuf, INPUT_SIZE * 4);
    else myReply("ERR get %d %d", c, n);
  } else if (!strncmp(s, "DEL ", 4)) {
    int c = -1, n = -1;
    sscanf(s + 4, "%d %d", &c, &n);
    String p;
    if (myNthSamplePath(c, n, &p) && SD.remove(p)) myReply("OK del %d %d", c, n);
    else myReply("ERR del %d %d", c, n);
  } else if (!strcmp(s, "TRAIN")) {
    myTrainCore();
  } else if (!strcmp(s, "CALIB")) {
#if MY_NORM_FROM_DATA
    myReplyCal();                                      // v006: the normalization belongs to the model; just report it
#else
    myCalibrate(true);
    myReplyCal();
#endif
  } else if (!strcmp(s, "GETMODEL")) {
    if (!myWeightsTrained) {
      myReply("ERR no trained model on the device yet");
    } else {
      myPackToBuf((float*)myBlobBuf);
      mySendBlob('M', 0, myBlobBuf, MY_PACKAGE_BYTES);
    }
  } else if (!strcmp(s, "INFER")) {
    if (!myWeightsTrained) {
      myReply("ERR no trained model on the device yet");
    } else {
      float w[INPUT_SIZE];
      float x[INPUT_SIZE];
      myCaptureWindow(w, false);
      memcpy(x, w, sizeof(w));
      myNormalizeInput(x);
      myForwardPass(x);
      int pred = 0;
      for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;
      char buf[MY_FRAME_MAX];
      int n = snprintf(buf, sizeof(buf), "RES %d", pred);
      for (int j = 0; j < NUM_CLASSES && n < (int)sizeof(buf) - 10; j++)
        n += snprintf(buf + n, sizeof(buf) - n, " %.4f", myFinal_output[j]);
      myReply("%s", buf);
      if (myLinkDebugOn) mySendBlob('S', 255, (const uint8_t*)w, INPUT_SIZE * 4);   // raw window for the parity check
    }
  } else {
    myReply("ERR unknown command: %s", s);
  }
  myBusy = false;
}

// Live ax,ay,az for the page while "debug frames" is on (the page re-sends DBG 1 every 5 s)
void myLinkHeartbeat() {
  unsigned long now = millis();
  if (myLinkDebugOn && now - myDebugLastMs > 15000) myLinkDebugOn = false;   // page went away
  if (myLinkDebugOn != myLinkWasDebug) {
    myLinkWasDebug = myLinkDebugOn;
    Serial.println(myLinkDebugOn ? "Debug frames ON" : "Debug frames OFF");
  }
  if (!myLinkDebugOn || myBusy || myRxActive || myReplyVia == 0) return;
  if (now - myLastHbMs < 250) return;
  myLastHbMs = now;
  float v[IMU_AXES];
  myReadAccel(v);
  myReply("HB %.3f %.3f %.3f", v[0], v[1], v[2]);
}


// ======================================================
// NOTE ON SENSOR FUSION EXTENSION
// ======================================================
// The 120-input vector is currently 40 x [ax, ay, az].
// To extend to other sensor combinations, change the layout here:
//
//   IMU_AXES = 6 -> [ax, ay, az, gx, gy, gz]   40 x 6 = 240  (update INPUT_SIZE = 240)
//   IMU_AXES = 2 -> [ax, ay]                    60 x 2 = 120  (adjust IMU_TIMESTEPS = 60)
//   Mixed sensors -> concatenate channels in the buffer, one entry per timestep
//
// (v003 keeps IMU_AXES = 3 because the page, the phone mapping and myReadAccel() all assume it.)
// ======================================================
