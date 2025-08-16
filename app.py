import gradio as gr
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from indic_unicode_mapper import IndicUnicodeMapper

MODEL_DIR = "tamil-lyrics-model"

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
mapper = IndicUnicodeMapper()
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

def generate_lyrics(theme: str) -> str:
    prompt = f"பாடல் தலைப்பு: {theme}\n"
    mapped = mapper.encode(prompt)
    input_ids = tokenizer(mapped, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )
    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return mapper.decode(raw)

iface = gr.Interface(
    fn=generate_lyrics,
    inputs=gr.Textbox(label="Theme"),
    outputs=gr.Textbox(label="Lyrics"),
    title="Tamil Lyrics Generator",
    description="Provide a theme to generate ~3 minute Tamil song lyrics.",
)

if __name__ == "__main__":
    iface.launch()
