# 🎤 Song Lyrics Generator

## Project Overview
This project uses a word-level LSTM trained on song lyrics to generate new text from a prompt. The model can create original song lyrics based on a short input phrase.

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Vocabulary Size | 4,981 |
| Embedding Dimension | 128 |
| Hidden Dimension | 512 |
| LSTM Layers | 2 |
| Dropout | 0.2 |
| Sequence Length | 20 |
| Total Parameters | ~ |

## Installation

```bash
git clone https://github.com/Prat-2005/song-lyrics-generator.git
cd lyrics-generator
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

## Example Outputs

**Prompt:** `love is`
> love is it out it get up go to get a little bit and i'm a man and a laugh like a punch at your neck and hit your ass and i don't know what to do this and i'll be livin' at me and let me bust the shit out the time i don't wanna see what i say i just want a lot i don't wanna seem to play but i ain't playin' i got a rhyme that i don't know what you say i'm just fed up i've been on my dick but i don't need no one i'm 'bout to be a father if i was just checking the same it's got a woman and a laugh just a couple of months and metaphors attached to my damn friend of mine and a place and smack a million bucks i'd get a list in the world

**Prompt:** `i am`
> i am no matter oh shit it's too mainstream and it's cold it's ridiculous he says but i'm going for that static that i was done with all i just do is murder hands through my head so i didn't feel so much more as a white life you can get ourselves off of this bitch and i got a breath and a sexy father i'ma make a whole damn life and i can't figure out which spice eminem i want to make a new wallet like a fat butt and make the beat down and turn off and make 'em all i do is number the way that i do is number time that you did i never want i think of a girl and i don't really got shit in my eyes but what would you do how to do

**Prompt:** `you better`
> you better run your whole ass in the door of me fuck all that was a girl all this shit is mine and i ain't gon' let 'em like that i won't do for her i just want it just stuck on

## Screenshots

![Screenshot](ss/image.png)

## Project Structure

```
lyrics-generator/
├── models/
├── notebooks/
├── src/
│   ├── model.py
│   ├── inference.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```