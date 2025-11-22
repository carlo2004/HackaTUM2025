import { CPRManager } from '../manager/CPRManager';
import { CPRStats } from '../model/Models';
import { SensorService } from '../services/SensorService';
import { common } from '@kit.AbilityKit';

@Entry
@Component
struct Index {
  private cprManager: CPRManager | undefined = undefined;
  private sensors: SensorService = new SensorService();

  @State feedbackMsg: string = "START";
  @State subText: string = "TAP TO BEGIN";
  @State isRunning: boolean = false;
  @State mainColor: string = '#E60000';
  @State currentBPM: string = "--";
  @State heartScale: number = 1.0;

  @State showCalibrate: boolean = false;

  aboutToAppear() {
    let context = getContext(this) as common.UIAbilityContext;
    this.cprManager = new CPRManager(context);

    this.cprManager.setUIUpdateCallback((stats: CPRStats, text: string) => {
      this.feedbackMsg = text;
      this.currentBPM = stats.currentBPM > 0 ? stats.currentBPM.toFixed(0) : "--";

      if (text.includes("PERFECT") || text.includes("READY")) {
        this.mainColor = '#00AA00';
        this.subText = text.includes("READY") ? "GO!" : "KEEP GOING";
      } else {
        this.mainColor = '#FF3333';
        if (text.includes("FAST")) this.subText = "SPEED UP";
        else if (text.includes("SLOW")) this.subText = "SLOW DOWN";
        else if (text.includes("DEEPER")) this.subText = "PUSH HARDER";
        else if (text.includes("RELEASE")) this.subText = "RELEASE CHEST";
        else this.subText = "CHECK RHYTHM";
      }
    });

    this.sensors.setCallback((bpm: number, depth: number, incompleteRecoil: boolean, mlFeedback: string) => {
      if (this.cprManager) {
        this.cprManager.ingestAIResults(bpm, depth, incompleteRecoil, mlFeedback);
      }
    });
  }

  toggleState() {
    if (!this.cprManager) return;

    this.isRunning = !this.isRunning;
    if (this.isRunning) {
      this.cprManager.startCPR(this.sensors);
      this.sensors.start();
      this.showCalibrate = true;
    } else {
      this.cprManager.stopCPR();
      this.sensors.stop();
      this.showCalibrate = false;
      this.feedbackMsg = "STOPPED";
      this.subText = "TAP TO RESTART";
      this.currentBPM = "--";
      this.mainColor = '#E60000';
    }
  }

  calibrate() {
    if (this.cprManager && this.isRunning) {
      this.cprManager.calibrateSensor(this.sensors);
    }
  }

  build() {
    Stack() {
      Circle()
        .width('80%')
        .height('80%')
        .fill(this.mainColor)
        .opacity(0.3)
        .scale({ x: this.heartScale, y: this.heartScale })
        .animation({
          duration: 545,
          curve: Curve.EaseInOut,
          iterations: -1,
          playMode: PlayMode.Alternate
        })
        .onAppear(() => { this.heartScale = 1.2; })

      Column() {
        if (!this.isRunning) {
          Blank().height('30%')
          Text("CPR ASSIST")
            .fontSize(24)
            .fontWeight(FontWeight.Bold)
            .fontColor(Color.White)
          Blank().height('5%')
          Button("START")
            .width(140).height(140)
            .borderRadius(70)
            .backgroundColor('#E60000')
            .fontColor(Color.White)
            .fontSize(24)
            .fontWeight(FontWeight.Bold)
            .onClick(() => this.toggleState())
          Blank().height('15%')
        } else {
          Blank().height('15%')

          Text(this.currentBPM)
            .fontSize(38)
            .fontColor(Color.White)
            .fontWeight(FontWeight.Bold)

          Text("BPM")
            .fontSize(14).fontColor(Color.Gray)
            .margin({ bottom: 10 })

          Text(this.feedbackMsg)
            .fontSize(28)
            .fontWeight(FontWeight.Bolder)
            .fontColor(Color.White)
            .textAlign(TextAlign.Center)
            .width('90%')

          Text(this.subText)
            .fontSize(20)
            .fontColor('#DDDDDD')
            .fontWeight(FontWeight.Medium)
            .margin({ top: 5 })

          Blank().layoutWeight(1)

          Row() {
            Button("STOP")
              .height(50)
              .width('35%')
              .backgroundColor('#330000')
              .fontColor('#FF6666')
              .borderRadius(12)
              .onClick(() => this.toggleState())

            Blank().width(12)

            if (this.showCalibrate) {
              Button("RE-CALIBRATE")
                .height(50)
                .layoutWeight(1)
                .backgroundColor('#333333')
                .fontColor(Color.White)
                .borderRadius(12)
                .onClick(() => this.calibrate())
            }
          }
          .width('90%')
          .margin({ bottom: 20 })
        }
      }
      .width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }
}