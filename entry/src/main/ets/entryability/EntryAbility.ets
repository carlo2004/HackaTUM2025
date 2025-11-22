import { AbilityConstant, ConfigurationConstant, UIAbility, Want, abilityAccessCtrl, Permissions } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';

const DOMAIN = 0x0000;

export default class EntryAbility extends UIAbility {

  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    this.context.getApplicationContext().setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
    hilog.info(DOMAIN, 'HACKATHON', '%{public}s', 'Ability onCreate');

    this.reqPermissionsFromUser();
  }

  async reqPermissionsFromUser() {
    let atManager = abilityAccessCtrl.createAtManager();

    let permissions: Array<Permissions> = [
      'ohos.permission.READ_HEALTH_DATA',
      'ohos.permission.ACCELEROMETER',
      'ohos.permission.GYROSCOPE',
      'ohos.permission.ACTIVITY_MOTION'
    ];

    try {

      atManager.requestPermissionsFromUser(this.context, permissions).then((data) => {
        hilog.info(DOMAIN, 'HACKATHON', 'Permission request result: ' + JSON.stringify(data));
      }).catch((err: object) => {
        hilog.error(DOMAIN, 'HACKATHON', 'Permission request failed: ' + JSON.stringify(err));
      });
    } catch (err) {
      hilog.error(DOMAIN, 'HACKATHON', 'Catch err: ' + JSON.stringify(err));
    }
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    hilog.info(DOMAIN, 'HACKATHON', '%{public}s', 'Ability onWindowStageCreate');

    let windowClass: window.Window | undefined = undefined;
    windowStage.getMainWindow((err, data) => {
      if (err.code) {
        hilog.error(DOMAIN, 'HACKATHON', 'Failed to obtain the main window.');
        return;
      }
      windowClass = data;
      windowClass.setWindowKeepScreenOn(true);
    });

    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(DOMAIN, 'HACKATHON', 'Failed to load the content.');
        return;
      }
      hilog.info(DOMAIN, 'HACKATHON', 'Succeeded in loading the content.');
    });
  }

  onWindowStageDestroy(): void { hilog.info(DOMAIN, 'HACKATHON', 'Ability onWindowStageDestroy'); }
  onForeground(): void { hilog.info(DOMAIN, 'HACKATHON', 'Ability onForeground'); }
  onBackground(): void { hilog.info(DOMAIN, 'HACKATHON', 'Ability onBackground'); }
  onDestroy(): void { hilog.info(DOMAIN, 'HACKATHON', 'Ability onDestroy'); }
}