# Load Testing

```bash
python load_test.py \
  --instance_type ml.g4dn.4xlarge \
  --tp_degree 1 \
  --vu 5 \
  --token $(cat ~/.cache/huggingface/token) \
  --endpoint_name stable-diffusion-endpoint \
  --endpoint_region us-east-2
```
