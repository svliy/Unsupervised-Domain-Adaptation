from description import describe_au
from model.clip import clip

if __name__ == '__main__':
    aus = [1]
    descriptions = describe_au(aus)
    for au, desc in zip(aus, descriptions):
        print(f"AU {au}: {desc}")