### Extract ImageBind Features

ImageBind官方地址：https://github.com/facebookresearch/ImageBind

使用之前，服务器需要自行下载权重文件。

可以使用ImageBind提取audio、visual和text等6种模态的特征，即可以同时提取6种，可以单独只提取其中一种模态特征。具体使用过程如下：

1. **数据预处理**

   1）声音：需要切分成目标长度，建议切成2秒的长度，采样率为16kHz；

   2）视频：按照需要的采样率对视频提帧并保存；

   3）文本：陈述句文本格式即可，无需特殊处理。

2. **数据存储**

   1）参数说明：

   ```python
   dir_audio_path：处理后的音频（段）存储路径
   dir_visual_path：处理后的视频帧存储路径
   dir_text_path：处理后的文本存储路径
   ----
   dst_audio_path：提取完的音频特征保存路径
   dst_visual_path：提取完的视频帧特征保存路径
   dst_text_path：提取完的文本特征保存路径
   ```

   2）目录结构：(视频名称样例：000001，000002)

   ```python
   dir_audio_path：
   |——000001
   |————000001.wav
   |————000002.wav
   |————000003.wav
   |————...
   |——000002
   |————000001.wav
   |————000002.wav
   |————000003.wav
   |————...
   -----
   dir_visual_path：
   |——000001
   |————000001.jpg
   |————000002.jpg
   |————000003.jpg
   |————...
   |——000002
   |————000001.jpg
   |————000002.jpg
   |————000003.jpg
   |————...
   -----
   dir_text_path
   ```

   3）保存格式：均为数组格式(.npy)，T为视频帧的数量或者音频的个数

   ```python
   audio features: [T, 1024]
   visual features: [T, 1024]
   text features: [1, 1024]
   ```

3. **执行**

   首先需要下载权重文件，下载地址：https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
   本程序默认存储路径：`.checkpoints/imagebind_huge.pth`，如有下载问题，欢迎邮件联系。

   ```python
   python extract_imagebind_feats.py
   ```

4. **其他**

   1）只提取单一模态特征（如只提取视频帧的特征）

   2）提取patch-level特征：只需要将./imagebind/models/helpers.py文件中的第124行注释，125行取消注释即可。

   ```python
   122 assert x.ndim >= 3
   123
   124 out = x[:, self.index, ...]     # frame-level features
   125 # out = x[:, self.index:, ...]  # patch-level features 
   126
   127 return out
   ```



如果使用过程中遇到任何问题，欢迎邮件交流，guangyaoli@ruc.edu.cn.