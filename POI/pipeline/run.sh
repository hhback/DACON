script="`readlink -f "${BASH_SOURCE[0]}"`"
HOMEDIR="`dirname "$script"`"

text_pretrained_model=roberta
image_pretrained_model=InceptionV3
batch_size=8
image_size=420
max_length=100

python main.py \
	--text_pretrained_model ${text_pretrained_model} \
        --image_pretrained_model ${image_pretrained_model} \
        --batch_size ${batch_size} \
        --image_size ${image_size} \
        --max_length ${max_length}