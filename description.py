def describe_au(aus):
    au_descriptions = {
        1: "The inner corners of the eyebrows are lifted slightly, "
           "the skin of the glabella and forehead above it is lifted slightly and wrinkles deepen slightly and a "
           "trace of new ones form in the center of the forehead.",
        2: "The outer part of the eyebrow raise is pronounced. The wrinkling above the right outer eyebrow "
           "has increased markedly, and the wrinkling on the left is pronounced. Increased exposure of the eye cover "
           "fold and skin is pronounced.",
        4: "Vertical wrinkles appear in the glabella and the eyebrows are pulled together. The inner parts of the "
           "eye-brows are pulled down a trace on the right and slightly on the left with traces of wrinkling at the corners.",
        6: "Lift your cheeks without actively raising up the lip corners. The infraorbital furrow has deepened "
           "slightly and bags or wrinkles under the eyes must increase. The infraorbital triangle is raised slightly.",
        7: "The lower eyelid is raised markedly and straightened slightly, causing slight bulging, and the narrowing "
           "of the eye aperture is marked to pronounced.",
        9: "Wrinkle the nose, draw skin on bridge of the nose upwards, lift the nasal wings up, raising the infraorbital "
           "triangle severely, and deepening the upper part of the nasolabial fold extremely as the upper lip is drawn up slightly.",
        10: "The center of upper lip is drawn straight up, the outer portions of upper lip are drawn up but not as high "
            "as the center. The infraorbital triangle is pushed up, the nasolabial furrow is deepened.",
        12: "The corners of the lips are markedly raised and angled up obliquely. The nasolabial furrow has deepened "
            "slightly and is raised obliquely slightly. The infraorbital triangle is raised slightly.",
        14: "The lip corners are extremely tightened, and the wrinkling as skin is pulled inwards around the lip corners "
            "is severe. The skin on the chin and lower lip is stretched towards the lip corners, and the lips are stretched "
            "and flattened against the teeth.",
        15: "The lip corners are pulled down slightly, with some lateral pulling and angling down of the corners, and "
            "slight bulges and wrinkles appear beyond the lip corners.",
        17: "The chin boss shows severe to extreme wrinkling as it is pushed up severely, and the lower lip is pushed up "
            "and out markedly.",
        23: "The lips are tightened maximally and the red parts are narrowed maximally, creating extreme wrinkling and "
            "bulging around the margins of the red parts of both lips.",
        24: "The lips are severely pressed together, severely bulging skin above and below the red parts, with severe "
            "narrowing of the lips and wrinkling above the upper lip.",
        25: "The teeth clearly show, and the lips are separated slightly. Nothing suggests that the jaw has dropped even "
            "though the upper teeth are not clearly visible.",
        26: "The jaw is lowered about as much as it can drop from relaxing of the muscles. The lips are parted to about "
            "the extent that the jaw lowering can produce."
    }

    return [au_descriptions.get(au, "No description available for this AU.") for au in aus]

def describe_au_all():
    au_descriptions = {
        1: "The inner corners of the eyebrows are lifted slightly, "
           "the skin of the glabella and forehead above it is lifted slightly and wrinkles deepen slightly and a "
           "trace of new ones form in the center of the forehead.",
        2: "The outer part of the eyebrow raise is pronounced. The wrinkling above the right outer eyebrow "
           "has increased markedly, and the wrinkling on the left is pronounced. Increased exposure of the eye cover "
           "fold and skin is pronounced.",
        4: "Vertical wrinkles appear in the glabella and the eyebrows are pulled together. The inner parts of the "
           "eye-brows are pulled down a trace on the right and slightly on the left with traces of wrinkling at the corners.",
        6: "Lift your cheeks without actively raising up the lip corners. The infraorbital furrow has deepened "
           "slightly and bags or wrinkles under the eyes must increase. The infraorbital triangle is raised slightly.",
        7: "The lower eyelid is raised markedly and straightened slightly, causing slight bulging, and the narrowing "
           "of the eye aperture is marked to pronounced.",
        9: "Wrinkle the nose, draw skin on bridge of the nose upwards, lift the nasal wings up, raising the infraorbital "
           "triangle severely, and deepening the upper part of the nasolabial fold extremely as the upper lip is drawn up slightly.",
        10: "The center of upper lip is drawn straight up, the outer portions of upper lip are drawn up but not as high "
            "as the center. The infraorbital triangle is pushed up, the nasolabial furrow is deepened.",
        12: "The corners of the lips are markedly raised and angled up obliquely. The nasolabial furrow has deepened "
            "slightly and is raised obliquely slightly. The infraorbital triangle is raised slightly.",
        14: "The lip corners are extremely tightened, and the wrinkling as skin is pulled inwards around the lip corners "
            "is severe. The skin on the chin and lower lip is stretched towards the lip corners, and the lips are stretched "
            "and flattened against the teeth.",
        15: "The lip corners are pulled down slightly, with some lateral pulling and angling down of the corners, and "
            "slight bulges and wrinkles appear beyond the lip corners.",
        17: "The chin boss shows severe to extreme wrinkling as it is pushed up severely, and the lower lip is pushed up "
            "and out markedly.",
        23: "The lips are tightened maximally and the red parts are narrowed maximally, creating extreme wrinkling and "
            "bulging around the margins of the red parts of both lips.",
        24: "The lips are severely pressed together, severely bulging skin above and below the red parts, with severe "
            "narrowing of the lips and wrinkling above the upper lip.",
        25: "The teeth clearly show, and the lips are separated slightly. Nothing suggests that the jaw has dropped even "
            "though the upper teeth are not clearly visible.",
        26: "The jaw is lowered about as much as it can drop from relaxing of the muscles. The lips are parted to about "
            "the extent that the jaw lowering can produce."
    }

    return au_descriptions

if __name__ == '__main__':
    au_des_list = describe_au_all()
    for k, v in au_des_list.items():
        print(f'{k}: {v}')