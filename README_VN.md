# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

Phiên bản VITS tùy chỉnh để phù hợp với các công nghệ mới.

- **Lưu ý**, nó chỉ là thử nghiệm, vẫn còn nhiều vấn đề cần khắc phục trước khi sử dụng.

## Có gì khác so với phiên bản thông thường:
- Được thiết kế để sử dụng với Torch 2.
- Sử dụng Accelerate.

## Sử dụng

Trước khi bắt đầu, bạn nên đảm bảo mình có:
- Cmake.
- espeak.

Đào tạo:
1. Đầu tiên, bạn hãy cài conda. Tôi khuyên bạn nên sử dụng nó để đơn giản hóa quá trình đào tạo.
2. Tạo một môi trường conda:
```bash 
conda create -n vits
```
3. Khởi động môi trường và cài đặt các requirements:
```bash 
conda activate vits
pip install -r requirements.txt
```

4. Hãy tạo một dataset cho riêng mình: 
- Bạn có thể tạo một dataset giống như cách bạn tạo dataset cho phiên bản vits thông thường.
- Hoặc bạn có thể tạo một custom dataset có cấu trúc như sau:
```
<you dataset>
|__ train
|   |__ audio1.wav
|   |__ audio1.txt.cleaned
|   |__ audio2.wav
|   |__ audio2.txt.cleaned
|   |__ *.wav
|   |__ *.txt.cleaned
|   |__ <folder>
|   |   |__*.wav
|   |   |__*.txt.cleaned
|
|__ eval
|   |__ audio1.wav
|   |__ audio1.txt.cleaned
|   |__ audio2.wav
|   |__ audio2.txt.cleaned
|   |__ *.wav
|   |__ *.txt.cleaned
|   |__ <folder>
|   |   |__*.wav
|   |   |__*.txt.cleaned
```
- Hoặc có thể là như thế này:
```
<you dataset>
|__ audio1.wav
|__ audio1.txt.cleaned
|__ audio2.wav
|__ audio2.txt.cleaned
|__ *.wav
|__ *.txt.cleaned
|__ <folder>
|   |__*.wav
|   |__*.txt.cleaned
```
- Còn nếu bạn lười thì có thể sử dụng preprocess cho nhanh cũng được. Sử dụng như sau:
```bash
python preprocess.py --data_dir <you_dataset> --save_to <save to> --src_lang <language of audio> --text_cleaners <text_cleaners>
```
5. Sau khi tất cả hoàn tất, hãy chạy lệnh sau để bắt đầu đào tạo:
```bash
accelerate launch train_accelerate.py 
    -c <config> 
    -m <save to>
    --custom_dataset=<you dataset>
    --batch_size=16 
    --cache_spectrogram_to_disk
    --cache_spectrogram
```

Ngồi chờ và thưởng thức. Đơn giản phải không.

## Vấn đề:
- Hiện tại vẫn chưa hỗi trợ train multi speakers, so-vits.
- Tốc độ train vẫn không có cải tiến, bù lại tiết kiệm bộ nhớ GPU một chút.
- Các vấn đề bug và lỗi, code chưa được tối ưu hóa.
- Thiếu một module nào đó.



