#!/usr/bin/python3

import requests
import xml.etree.ElementTree as ET
import os
import time
import re
import json
from tts_say import say

response = requests.get("http://wthrcdn.etouch.cn/WeatherApi?city=%E5%A4%A9%E6%B4%A5")
# print(response.text)
tree = ET.fromstring(response.text)

def get_text(s):
    globals()[s.replace('/', '_').replace('[', '').replace(']', '')] = tree.find(s).text

# wendu = tree.find('wendu').text
# fengxiang = tree.find('fengxiang').text

get_text('wendu')
get_text('fengxiang')
get_text('fengli')
get_text('shidu')
get_text('environment/pm25')
get_text('environment/quality')
get_text('forecast/weather/high')
get_text('forecast/weather/low')
get_text('forecast/weather/day/type')
get_text('zhishus/zhishu[11]/detail')

result_tq = r'今天是%s。现在外面有%s的%s,户外温度:%s度。空气质量:%s,PM2.5：%s。今天白天的天气:%s。%s' % \
    (time.strftime('%m月%d日', time.localtime()), \
    fengli, fengxiang, wendu, environment_quality, environment_pm25, forecast_weather_day_type, zhishus_zhishu11_detail)

print(result_tq)
say(result_tq)

