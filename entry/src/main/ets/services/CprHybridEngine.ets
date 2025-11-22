/**
 * CprHybridEngine.ts
 * The brain that converts Raw Data -> Features -> AI Decision.
 */

export interface CprFeedback {
  bpm: number;
  speedScore: number; // 0-100%
  statusMessage: string;
  uiColor: string; // Hex code
}

export class CprHybridEngine {

  // --- MAIN ENTRY POINT ---
  public static analyze(
      xValues: number[], 
      yValues: number[], 
      zValues: number[], 
      gyroXValues: number[],
      rescuerHr: number
  ): CprFeedback {
    
    // 1. FEATURE ENGINEERING (Matches Python Logic)

    // A. TILT (Mean X/Y) - Checks for 90 degree angle
    const meanX = CprHybridEngine.calculateMean(xValues);
    const meanY = CprHybridEngine.calculateMean(yValues);
    const tiltError = Math.sqrt((meanX * meanX) + (meanY * meanY));

    // B. WOBBLE (Gyro StdDev) - Checks for rocking
    let wobbleScore = 0;
    if (gyroXValues.length > 1) {
        wobbleScore = CprHybridEngine.calculateStdDev(gyroXValues);
    }

    // C. DEPTH (Accel Z StdDev) - Checks for push strength
    const stdDevZ = CprHybridEngine.calculateStdDev(zValues);

    // D. RECOIL (Accel Z Max) - Checks for leaning
    // Note: Gravity is negative (-9.8). Max value is closest to 0 (recoil point).
    const recoilProxy = Math.max(...zValues);

    // 2. CALCULATE BPM (Simple Math)
    // Zero-Crossing method: Count how many times we cross the average line
    const meanZ = CprHybridEngine.calculateMean(zValues);
    let crossings = 0;
    for (let i = 1; i < zValues.length; i++) {
       if ((zValues[i] > meanZ && zValues[i-1] < meanZ) || 
           (zValues[i] < meanZ && zValues[i-1] > meanZ)) {
           crossings++;
       }
    }
    const bpm = (crossings / 2) * 60;

    // 3. CALCULATE SPEED SCORE (Math)
    // 110 is perfect. Deduct points for being off.
    const dist = Math.abs(110 - bpm);
    let speedScore = 100 - (dist * 2); 
    speedScore = Math.max(0, Math.min(100, speedScore));

    // 4. RUN AI MODEL (Decision Tree)
    // Feature Vector: [std_dev_z, recoil, hr, tilt, wobble]
    const inputVector = [stdDevZ, recoilProxy, rescuerHr, tiltError, wobbleScore];
    const aiClass = CprHybridEngine.runDecisionTree(inputVector);

    // 5. GENERATE FEEDBACK
    // Priority: Tired > Leaning > Angle > Speed
    
    if (aiClass === 2) {
      return { bpm, speedScore, statusMessage: "TIRED: SWAP RESCUER", uiColor: "#FF0000" }; // Red
    }
    if (aiClass === 1) {
      return { bpm, speedScore, statusMessage: "RELEASE CHEST FULLY", uiColor: "#0000FF" }; // Blue
    }
    if (aiClass === 3) {
       return { bpm, speedScore, statusMessage: "LOCK ARMS VERTICAL", uiColor: "#FFA500" }; // Orange
    }
    if (aiClass === 4) {
       return { bpm, speedScore, statusMessage: "PUSH HARDER!", uiColor: "#FFA500" }; // Orange
    }

    // Class 0 (Good Form) -> Check Speed
    if (speedScore > 85) {
      return { bpm, speedScore, statusMessage: "PERFECT RHYTHM", uiColor: "#00FF00" }; // Green
    } else if (bpm > 110) {
      return { bpm, speedScore, statusMessage: "TOO FAST", uiColor: "#FFA500" };
    } else {
      return { bpm, speedScore, statusMessage: "PUSH FASTER", uiColor: "#FFA500" };
    }
  }

  // --- THE AI PLACEHOLDER ---
  // You will paste the output of 'train_cpr_final.py' here later!
  private static runDecisionTree(input: number[]): number {
      // Example Logic (Delete this once you have the real model):
      // input[0] is stdDevZ. input[2] is HR.
      
      if (input[0] < 3.0) return 4; // Weak
      if (input[1] > -5.0) return 1; // Leaning (Example threshold)
      
      return 0; // Default to GOOD
  }

  // --- MATH HELPERS ---
  private static calculateMean(data: number[]): number {
    if (data.length === 0) return 0;
    return data.reduce((a, b) => a + b, 0) / data.length;
  }

  private static calculateStdDev(data: number[]): number {
    if (data.length < 2) return 0;
    const mean = CprHybridEngine.calculateMean(data);
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    return Math.sqrt(variance);
  }
}