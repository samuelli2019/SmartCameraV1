from subprocess import Popen, PIPE
from baidu_tts import BaiduTTS


def get_stream(sentence):
    tts = BaiduTTS('', '')
    return tts.say(sentence).read()

def say_sync(sentence):
    sound_buffer = get_stream(sentence)
    player = Popen(['/usr/bin/mpg123', '-'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    player.stdin.write(sound_buffer)
    player.stdin.flush()
    player.stdin.close()
    return player.wait()

def say_async(sentence):
    sound_buffer = get_stream(sentence)
    player = Popen(['/usr/bin/mpg123', '-'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    player.stdin.write(sound_buffer)
    player.stdin.flush()
    player.stdin.close()
    return player.pid

say = say_sync

if __name__ == '__main__':
    import sys
    s = ' '.join(sys.argv[1:])
    content = get_stream(s)
    with open('result.mp3', 'wb') as f:
        f.write(content)
