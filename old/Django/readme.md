# Django+flask+fastapi

## 1.资料

[04-快速使用django展示数据_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1nNr7YZESq?spm_id_from=333.788.player.switch&vd_source=8b69015a784e94f6a869001308d33fa5&p=4)

【2025版-零基础玩转Python Django5项目实战-学完可就业】 https://www.bilibili.com/video/BV1N1421U76L/?p=5&share_source=copy_web&vd_source=b1560a316ec9a486cde3dbbfef0ffd0f

## 2.笔记

原生开发：不利用现有框架，从0开始

敏捷开发：使用框架

二次开发：

![image-20251028153109320](./pic/image-20251028153109320.png)

![image-20251028160541982](./pic/image-20251028160541982.png)

![image-20251028160811117](./pic/image-20251028160811117.png)

![image-20251218100155670](./pic/image-20251218100155670.png)

![image-20251028161643568](./pic/image-20251028161643568.png)

![image-20251028162647755](./pic/image-20251028162647755.png)

![image-20251028162716603](./pic/image-20251028162716603.png)

telnet curl工具

postman apifox

网络知识http协议

![image-20251028162738253](./pic/image-20251028162738253.png)

![image-20251028162826540](./pic/image-20251028162826540.png)

![image-20251028162844193](./pic/image-20251028162844193.png)

![image-20251028162922541](./pic/image-20251028162922541.png)

put和patch

![image-20251028163543636](./pic/image-20251028163543636.png)

OA系统开发自动化办公开发

![image-20251028163716092](./pic/image-20251028163716092.png)

协议升级http——>websocket

![image-20251028163905097](./pic/image-20251028163905097.png)

![image-20251028164050023](./pic/image-20251028164050023.png)

![image-20251028164633836](./pic/image-20251028164633836.png)

http 80端口转向https 443端口

多数情况下304有益 304使用本地缓存但是又更新情况下

阿里郎代码监控

pragma 

请求头

![image-20251028164751790](./pic/image-20251028164751790.png)

网页静态化 动态化

![image-20251028165718914](./pic/image-20251028165718914.png)

![image-20251028171546806](./pic/image-20251028171546806.png)

![image-20251028171911713](./pic/image-20251028171911713.png)

wsgi协议

![image-20251028172151732](./pic/image-20251028172151732.png)

![image-20251028172622561](./pic/image-20251028172622561.png)

![image-20251028172928151](./pic/image-20251028172928151.png)

![image-20251028173007093](./pic/image-20251028173007093.png)

**![image-20251028173737859](./pic/image-20251028173737859.png)**

![image-20251029113256407](./pic/image-20251029113256407.png)

![image-20251029113315138](./pic/image-20251029113315138.png)

![image-20251029113540901](./pic/image-20251029113540901.png)

![image-20251029113927683](./pic/image-20251029113927683.png)

request.GET[‘pwd’] 如果没写入会报错，速度高但是没高多少

request.GET.get(‘pwd’) 如果没写入就是none

![image-20251029114322812](./pic/image-20251029114322812.png)

![image-20251029115125208](./pic/image-20251029115125208.png)

常见的请求头

![image-20251030095953076](./pic/image-20251030095953076.png)

![image-20251030100455882](./pic/image-20251030100455882.png)

![image-20251030100612081](./pic/image-20251030100612081.png)

![image-20251030102554949](./pic/image-20251030102554949.png)

![image-20251030103156783](./pic/image-20251030103156783.png)

![image-20251030103415499](./pic/image-20251030103415499.png)

![image-20251030103555762](./pic/image-20251030103555762.png)

MIME查看文件content格式

![image-20251030105309492](./pic/image-20251030105309492.png)

![image-20251030110446230](./pic/image-20251030110446230.png)

![image-20251030110854814](./pic/image-20251030110854814.png)

![image-20251030111430438](./pic/image-20251030111430438.png)

![image-20251030111604559](./pic/image-20251030111604559.png)

url地址记录用户身份实现多个qq同时登录早期增加用户量目的

![image-20251030113317118](./pic/image-20251030113317118.png)

![image-20251030113759866](./pic/image-20251030113759866.png)

itsdangerous

![image-20251030135229326](./pic/image-20251030135229326.png)

![image-20251030135251467](./pic/image-20251030135251467.png)

![image-20251030141328846](./pic/image-20251030141328846.png)

base64 atob btoa

![image-20251030141817360](./pic/image-20251030141817360.png)

path

![image-20251030142125492](./pic/image-20251030142125492.png)

![image-20251030142959025](./pic/image-20251030142959025.png)

![image-20251030143305743](./pic/image-20251030143305743.png)

![image-20251030150550650](./pic/image-20251030150550650.png)

str在前面会覆盖调uuid

# fastapi

【黑马程序员PythonWeb开发：FastAPI从入门到实战视频教程，涵盖路由、依赖注入、Pydantic、异步编程、ORM、项目拆分、模型训练、部署、接口测试】 https://www.bilibili.com/video/BV1zV2QBtE39/?p=2&share_source=copy_web&vd_source=b1560a316ec9a486cde3dbbfef0ffd0f

![image-20251210170840085](./pic/image-20251210170840085.png)

![image-20251210170915341](./pic/image-20251210170915341.png)

![image-20251210170925238](./pic/image-20251210170925238.png)

![image-20251210170958465](./pic/image-20251210170958465.png)

直接改成docs就有测试区域

![image-20251210172130778](./pic/image-20251210172130778.png)

![image-20251210172234278](./pic/image-20251210172234278.png)

![image-20251210172249553](./pic/image-20251210172249553.png)

![image-20251210173530384](./pic/image-20251210173530384.png)







