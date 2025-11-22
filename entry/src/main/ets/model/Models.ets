export class PatientProfile {
  isChild: boolean = false;
  targetBPM: number = 110;
}

export class CPRStats {
  startTime: number = 0;
  totalCompressions: number = 0;
  currentBPM: number = 0;
  lastPushDepth: number = 0;
}