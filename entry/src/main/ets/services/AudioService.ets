import { media } from '@kit.MediaKit';
import { common } from '@kit.AbilityKit';
import { audio } from '@kit.AudioKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

export class AudioService {
  private avPlayer: media.AVPlayer | undefined;
  private context: common.UIAbilityContext;

  constructor(context: common.UIAbilityContext) {
    this.context = context;
  }

  async playVoiceCommand(filename: string) {

    if (this.avPlayer) {
      try { await this.avPlayer.release(); } catch (e) {}
      this.avPlayer = undefined;
    }

    try {

      this.avPlayer = await media.createAVPlayer();

      if (!this.context || !this.context.resourceManager) {
        hilog.error(0x0000, 'HACKATHON', 'Context invalid');
        return;
      }
      let rawFd = await this.context.resourceManager.getRawFd(filename);
      hilog.info(0x0000, 'HACKATHON', `Loading: ${filename}`);

      this.avPlayer.fdSrc = {
        fd: rawFd.fd,
        offset: rawFd.offset,
        length: rawFd.length
      };

      this.avPlayer.on('stateChange', async (state) => {
        hilog.info(0x0000, 'HACKATHON', `State: ${state}`);

        switch (state) {
          case 'initialized':

            if (this.avPlayer) {
              this.avPlayer.audioRendererInfo = {
                content: audio.ContentType.CONTENT_TYPE_SPEECH,
                usage: audio.StreamUsage.STREAM_USAGE_MEDIA,
                rendererFlags: 0
              };

              await this.avPlayer.prepare();
            }
            break;

          case 'prepared':

            if (this.avPlayer) {
              this.avPlayer.setVolume(1.0);
              await this.avPlayer.play();
            }
            break;

          case 'completed':
            hilog.info(0x0000, 'HACKATHON', 'Audio Finished');
            await this.avPlayer?.release();
            this.avPlayer = undefined;
            break;
        }
      });

      this.avPlayer.on('error', (err: BusinessError) => {
        hilog.error(0x0000, 'HACKATHON', `Player Error: ${err.code} - ${err.message}`);
        this.avPlayer?.release();
      });

    } catch (e) {
      hilog.error(0x0000, 'HACKATHON', `Exception: ${JSON.stringify(e)}`);
    }
  }
}