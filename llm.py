from groq import Groq

with open('api.txt','r') as f:
    s = f.read()
    client = Groq(api_key=s)

data = '[{"禁止左轉",30,100"\}, {"7-9":"30,110\}]'
text_message = f'我需要生成一個 Fine Tune 資料，是從圖片辨識出路牌及相關資訊，接著你要從這些資訊給駕駛簡短的資訊，我獲得的資料格式如下，前面是號誌指示，後面兩個數字代表的是座標：\n{data}\n請幫我直接以繁體中文總結出一項指示在你最後的結果，並以【】包起來'
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_message},
            ],
        }
    ],
    model="llama3-70b-8192",
)


print(response.choices[0].message.content)