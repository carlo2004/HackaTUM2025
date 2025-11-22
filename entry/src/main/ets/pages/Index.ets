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
  @State heartScale: number = 1.0;
  @State mainColor: string = '#E60000';
  @State currentBPM: string = "--";

  @State isGuiding: boolean = false;

  aboutToAppear() {
    let context = getContext(this) as common.UIAbilityContext;
    this.cprManager = new CPRManager(context);

    this.cprManager.setUIUpdateCallback((stats: CPRStats, text: string) => {

      if (this.isGuiding) return;

      this.feedbackMsg = text;
      this.currentBPM = stats.currentBPM > 0 ? stats.currentBPM.toFixed(0) : "--";

      if (text.includes("PERFECT")) {
        this.mainColor = '#00AA00';
        this.subText = "KEEP GOING";
      } else {
        this.mainColor = '#FF3333';

        if (text.includes("FAST")) this.subText = "SLOW DOWN";
        else if (text.includes("SLOW")) this.subText = "SPEED UP";
        else if (text.includes("HARDER")) this.subText = "PUSH DEEPER";
        else if (text.includes("SOFTER")) this.subText = "LESS FORCE";
        else this.subText = "CHECK FORM";
      }
    });

    this.sensors.setCallback((bpm: number, depth: number) => {
      if (this.cprManager) {
        this.cprManager.ingestAIResults(bpm, depth);
      }
    });
  }

  runGuideSequence() {
    this.isGuiding = true;

    setTimeout(() => {
      if (!this.isRunning) return;

      this.isGuiding = false;
      this.feedbackMsg = "ANALYZING...";
      this.subText = "PUSH HARD & FAST";
    }, 14000);
  }

  toggleState() {
    if (!this.cprManager) return;

    this.isRunning = !this.isRunning;
    if (this.isRunning) {
      this.cprManager.startCPR();
      this.sensors.start();

      this.runGuideSequence();

    } else {
      this.cprManager.stopCPR();
      this.sensors.stop();

      this.isGuiding = false;
      this.feedbackMsg = "STOPPED";
      this.subText = "TAP TO RESTART";
      this.currentBPM = "--";
      this.mainColor = '#E60000';
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
        .onAppear(() => {
          this.heartScale = 1.2;
        })

      Column() {

        if (this.isGuiding) {

          Blank().height('10%')

          Image($r('app.media.cpr_hands'))
            .width('85%')
            .objectFit(ImageFit.Contain)
            .layoutWeight(1)
            .shadow({ radius: 15, color: 'rgba(0,0,0,0.5)' })

          Text("FOLLOW AUDIO")
            .fontSize(14)
            .fontColor('#AAAAAA')
            .fontWeight(FontWeight.Medium)
            .margin({ top: 10, bottom: 10 })

        } else {

          Blank().height('15%')

          Text(this.currentBPM)
            .fontSize(24)
            .fontColor(Color.Gray)
            .fontWeight(FontWeight.Bold)
            .margin({ bottom: 5 })

          Text(this.feedbackMsg)
            .fontSize(32)
            .minFontSize(10)
            .maxFontSize(32)
            .maxLines(1)
            .fontWeight(FontWeight.Bolder)
            .fontColor(Color.White)
            .textAlign(TextAlign.Center)
            .width('90%')
            .height(40)

          Text(this.subText)
            .fontSize(18)
            .fontColor('#DDDDDD')
            .fontWeight(FontWeight.Medium)
            .textAlign(TextAlign.Center)
            .width('95%')
            .margin({ top: 5 })

          Blank().height('15%')
        }

        Button(this.isRunning ? "STOP" : "START")
          .width(120)
          .height(45)
          .backgroundColor(Color.White)
          .fontColor(Color.Black)
          .fontWeight(FontWeight.Bold)
          .fontSize(18)
          .type(ButtonType.Capsule)
          .shadow({ radius: 10, color: '#40000000', offsetY: 5 })
          .margin({ bottom: 20 })
          .onClick(() => {
            this.toggleState();
          })
      }
      .width('100%')
      .height('100%')
      .justifyContent(FlexAlign.Center)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }
}