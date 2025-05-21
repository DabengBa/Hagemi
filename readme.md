在原项目基础上进行自用化改造：

1. 支持多模态信息（例如发送图片）
2. 调整日志内容
   1. 显示KEY的后8位
   2. 增加时间戳（强制设置为北京时间）
3. 增加MAX_RETRY变量（默认为3）
   说明：原项目的重试次数=KEY数量，某些时候这个重试次数太多了没有必要。现在会在MAX_RETRY和KEY数量中选择较小的那个
   比如MAX_RETRY为3，但你只有2个KEY，那么重试次数为2
4. 记录429错误时间(达到速率上线)吗冷却4小时. 并在主页显示上次429的时间.
5. 还有些其他的微调我不记得了。

其他信息请参考原项目。

简单说下HUGGING FACE部署：

1. 新建space空间，选择docker，空白
2. 上传文件（切记不要上传readme.md）
3. 在Setting中配置SECREATS
   1. PASSWARD, 默认 123，这是使用API时候的密钥
   2. GEMINI_API_KEYS，例如 Asyz1...., Asyz2....，这是API KEY，使用逗号分割。
4. 在Setting中配置Variant （可选）
   1. MAX_RETRY，默认 3
