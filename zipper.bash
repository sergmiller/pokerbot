#!/bin/bash
bot=$1
len=${#bot}
bot_name=${bot:5:len-8}
image="sberbank/python"
entry_point="python $bot"
echo -e "{\"image\":\""$image"\", \"entry_point\":\""$entry_point"\"}" > metadata.json
zip zips/$bot_name.zip $bot metadata.json
