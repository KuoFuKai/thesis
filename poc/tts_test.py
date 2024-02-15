import time
from tts_util import say

if __name__ == "__main__":
    say("Hello, this is the first message.")
    time.sleep(2)
    say("And this is a second message, after stopping the first one.")
    time.sleep(2)
    say("And this is a third message, after stopping the second one.")