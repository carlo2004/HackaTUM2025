// src/main/ets/model/Models.ets

// Defines the patient profile (loaded from offline file)
export class PatientProfile {
  isChild: boolean = false;
  targetBPM: number = 110; // Default 110
}

// Defines the live session stats
export class CPRStats {
  startTime: number = 0;
  totalCompressions: number = 0;
  currentBPM: number = 0;
  lastPushDepth: number = 0; // Estimated from Z-force
}