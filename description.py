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

def neg_describe_au(aus):
    au_descriptions = {
        1: ["The inner corners of the eyebrows are lowered slightly, the skin of the glabella and forehead above it is relaxed slightly, and wrinkles smooth out slightly, while a trace of old ones fades in the center of the forehead.",
            "The inner corners of the eyebrows are pulled slightly downwards, the skin of the glabella and forehead above it is softened slightly, and wrinkles lessen slightly, while a trace of new lines is barely visible in the center of the forehead.",
            "The inner corners of the eyebrows are dropped slightly, the skin of the glabella and forehead above it is relaxed slightly, and wrinkles fade slightly, while a trace of existing ones lightens in the center of the forehead.",
            "The inner corners of the eyebrows are flattened slightly, the skin of the glabella and forehead above it is softened slightly, and wrinkles ease slightly, while a trace of old ones disappears in the center of the forehead.",
            "The inner corners of the eyebrows are pressed gently down, the skin of the glabella and forehead above it is smoothed slightly, and wrinkles diminish slightly, while a trace of fine lines vanishes in the center of the forehead.",
            "The inner corners of the eyebrows are relaxed slightly, the skin of the glabella and forehead above it is loosened slightly, and wrinkles lessen slightly, while a trace of new ones disappears in the center of the forehead.",
            "The inner corners of the eyebrows are tilted slightly downwards, the skin of the glabella and forehead above it is smoothed slightly, and wrinkles ease slightly, while a trace of old ones fades in the center of the forehead.",
            "The inner corners of the eyebrows are tugged slightly down, the skin of the glabella and forehead above it is tightened slightly, and wrinkles diminish slightly, while a trace of existing ones fades in the center of the forehead.",
            "The inner corners of the eyebrows are turned slightly downwards, the skin of the glabella and forehead above it is relaxed slightly, and wrinkles grow slightly, while a trace of fine lines diminishes in the center of the forehead.",
            "The inner corners of the eyebrows are bent slightly down, the skin of the glabella and forehead above it is smoothed slightly, and wrinkles reduce slightly, while a trace of new ones disappears in the center of the forehead."],
        2: ["The outer part of the eyebrow raise is diminished. The wrinkling above the right outer eyebrow has decreased markedly, and the wrinkling on the left is less pronounced. Decreased exposure of the eye cover fold and skin is subtle.",
            "The outer part of the eyebrow raise is lowered. The wrinkling above the right outer eyebrow has smoothed out markedly, and the wrinkling on the left is less pronounced. Reduced exposure of the eye cover fold and skin is less noticeable.",
            "The outer part of the eyebrow raise is less pronounced. The wrinkling above the right outer eyebrow has lessened markedly, and the wrinkling on the left is less evident. Diminished exposure of the eye cover fold and skin is slight.",
            "The outer part of the eyebrow raise is relaxed. The wrinkling above the right outer eyebrow has flattened markedly, and the wrinkling on the left is softer. Less exposure of the eye cover fold and skin is subtle.",
            "The outer part of the eyebrow raise is softened. The wrinkling above the right outer eyebrow has receded markedly, and the wrinkling on the left is faint. Diminished exposure of the eye cover fold and skin is minimal.",
            "The outer part of the eyebrow raise is toned down. The wrinkling above the right outer eyebrow has diminished markedly, and the wrinkling on the left is less pronounced. Reduced exposure of the eye cover fold and skin is subtle.",
            "The outer part of the eyebrow raise is lowered slightly. The wrinkling above the right outer eyebrow has eased markedly, and the wrinkling on the left is softened. Less exposure of the eye cover fold and skin is reduced.",
            "The outer part of the eyebrow raise is weak. The wrinkling above the right outer eyebrow has smoothed noticeably, and the wrinkling on the left is subtle. Decreased exposure of the eye cover fold and skin is slight.",
            "The outer part of the eyebrow raise is diminished significantly. The wrinkling above the right outer eyebrow has eased up markedly, and the wrinkling on the left is less prominent. Reduced exposure of the eye cover fold and skin is minimal.",
            "The outer part of the eyebrow raise is reduced. The wrinkling above the right outer eyebrow has subsided markedly, and the wrinkling on the left is less defined. Less exposure of the eye cover fold and skin is barely visible."],
        4: ["Vertical wrinkles fade in the glabella, and the eyebrows are relaxed. The inner parts of the eyebrows are raised a trace on the right and slightly on the left, with smooth skin at the corners.",
            "Vertical wrinkles diminish in the glabella, and the eyebrows are separated. The inner parts of the eyebrows are lifted a trace on the right and slightly on the left, with no wrinkling at the corners.",
            "Vertical wrinkles lessen in the glabella, and the eyebrows are opened. The inner parts of the eyebrows are raised slightly on the right and a trace on the left, with clear skin at the corners.",
            "Vertical wrinkles smooth out in the glabella, and the eyebrows are spread apart. The inner parts of the eyebrows are elevated a trace on the right and slightly on the left, with smoothness at the corners.",
            "Vertical wrinkles reduce in the glabella, and the eyebrows are relaxed. The inner parts of the eyebrows are pushed up a trace on the right and slightly on the left, with no visible wrinkling at the corners.",
            "Vertical wrinkles disappear in the glabella, and the eyebrows are opened. The inner parts of the eyebrows are lifted slightly on the right and a trace on the left, with flat skin at the corners.",
            "Vertical wrinkles flatten in the glabella, and the eyebrows are apart. The inner parts of the eyebrows are raised a trace on the right and slightly on the left, with no traces of wrinkling at the corners.",
            "Vertical wrinkles vanish in the glabella, and the eyebrows are relaxed. The inner parts of the eyebrows are elevated a trace on the right and slightly on the left, with smooth skin at the corners.",
            "Vertical wrinkles subside in the glabella, and the eyebrows are eased. The inner parts of the eyebrows are pushed up slightly on the right and a trace on the left, with clear skin at the corners.",
            "Vertical wrinkles disappear in the glabella, and the eyebrows are opened widely. The inner parts of the eyebrows are lifted a trace on the right and slightly on the left, with no signs of wrinkling at the corners."],
        6: ["Lower your cheeks without actively raising the lip corners. The infraorbital furrow has softened slightly, and bags or wrinkles under the eyes must decrease. The infraorbital triangle is lowered slightly.",
            "Relax your cheeks without actively pulling down the lip corners. The infraorbital furrow has smoothed slightly, and bags or wrinkles under the eyes must lessen. The infraorbital triangle is dropped slightly.",
            "Drop your cheeks without actively lifting the lip corners. The infraorbital furrow has faded slightly, and bags or wrinkles under the eyes must shrink. The infraorbital triangle is diminished slightly.",
            "Soften your cheeks without actively tightening the lip corners. The infraorbital furrow has relaxed slightly, and bags or wrinkles under the eyes must reduce. The infraorbital triangle is lowered slightly.",
            "Sink your cheeks without actively elevating the lip corners. The infraorbital furrow has eased slightly, and bags or wrinkles under the eyes must lighten. The infraorbital triangle is flattened slightly.",
            "Lower your cheeks without actively raising the lip corners. The infraorbital furrow has flattened slightly, and bags or wrinkles under the eyes must diminish. The infraorbital triangle is dropped slightly.",
            "Depress your cheeks without actively elevating the lip corners. The infraorbital furrow has lessened slightly, and bags or wrinkles under the eyes must subside. The infraorbital triangle is lowered slightly.",
            "Relax your cheeks without actively lifting the lip corners. The infraorbital furrow has faded slightly, and bags or wrinkles under the eyes must decrease. The infraorbital triangle is eased slightly.",
            "Droop your cheeks without actively raising the lip corners. The infraorbital furrow has smoothed slightly, and bags or wrinkles under the eyes must reduce. The infraorbital triangle is lowered slightly.",
            "Ease your cheeks without actively raising the lip corners. The infraorbital furrow has softened slightly, and bags or wrinkles under the eyes must shrink. The infraorbital triangle is dropped slightly."],
        7: ["The lower eyelid is lowered markedly and relaxed slightly, causing slight flattening, and the widening of the eye aperture is marked to pronounced.",
            "The lower eyelid is dropped markedly and softened slightly, causing slight recession, and the broadening of the eye aperture is marked to pronounced.",
            "The lower eyelid is depressed markedly and eased slightly, causing slight withdrawal, and the increase in the eye aperture is marked to pronounced.",
            "The lower eyelid is relaxed markedly and sagged slightly, causing slight smoothness, and the expansion of the eye aperture is marked to pronounced.",
            "The lower eyelid is sagged significantly and slackened slightly, causing slight retraction, and the enlarging of the eye aperture is marked to pronounced.",
            "The lower eyelid is lowered markedly and drooped slightly, causing slight subsiding, and the widening of the eye aperture is marked to pronounced.",
            "The lower eyelid is depressed greatly and softened slightly, causing slight diminishing, and the broadening of the eye aperture is marked to pronounced.",
            "The lower eyelid is dropped significantly and relaxed slightly, causing slight disappearance, and the opening of the eye aperture is marked to pronounced.",
            "The lower eyelid is lowered markedly and slackened slightly, causing slight retracting, and the expanding of the eye aperture is marked to pronounced.",
            "The lower eyelid is sagged markedly and softened slightly, causing slight flattening, and the widening of the eye aperture is marked to pronounced."],
        9: ["Smooth the nose, draw skin on the bridge of the nose downwards, lower the nasal wings, dropping the infraorbital triangle severely, and shallowing the upper part of the nasolabial fold extremely as the upper lip is relaxed slightly.",
            "Unwrinkle the nose, pull skin on the bridge of the nose down, depress the nasal wings, lowering the infraorbital triangle severely, and lessening the upper part of the nasolabial fold extremely as the upper lip is released slightly.",
            "Relax the nose, draw skin on the bridge of the nose down, lower the nasal wings, dropping the infraorbital triangle severely, and softening the upper part of the nasolabial fold extremely as the upper lip is depressed slightly.",
            "Flatten the nose, pull skin on the bridge of the nose downward, lower the nasal wings, dropping the infraorbital triangle severely, and shallowing the upper part of the nasolabial fold extremely as the upper lip is lowered slightly.",
            "Ease the nose, draw skin on the bridge of the nose downwards, depress the nasal wings, dropping the infraorbital triangle severely, and lightening the upper part of the nasolabial fold extremely as the upper lip is let down slightly.",
            "Calm the nose, pull skin on the bridge of the nose down, sink the nasal wings, dropping the infraorbital triangle severely, and softening the upper part of the nasolabial fold extremely as the upper lip is lowered slightly.",
            "Unwrinkle the nose, draw skin on the bridge of the nose down, decrease the elevation of the nasal wings, dropping the infraorbital triangle severely, and diminishing the upper part of the nasolabial fold extremely as the upper lip is relaxed slightly.",
            "Flatten the nose, draw skin on the bridge of the nose downward, lower the nasal wings, dropping the infraorbital triangle severely, and lessening the upper part of the nasolabial fold extremely as the upper lip is let down slightly.",
            "Ease the nose, draw skin on the bridge of the nose downwards, depress the nasal wings, dropping the infraorbital triangle severely, and softening the upper part of the nasolabial fold extremely as the upper lip is lowered slightly.",
            "Relax the nose, draw skin on the bridge of the nose down, lower the nasal wings, dropping the infraorbital triangle severely, and shallowing the upper part of the nasolabial fold extremely as the upper lip is depressed slightly."],
        10: ["The center of the upper lip is drawn straight down, the outer portions of the upper lip are drawn down but not as low as the center. The infraorbital triangle is pulled down, the nasolabial furrow is shallowed.",
            "The center of the upper lip is pulled straight down, the outer portions of the upper lip are lowered but not as low as the center. The infraorbital triangle is depressed, the nasolabial furrow is lessened.",
            "The center of the upper lip is relaxed downwards, the outer portions of the upper lip are lowered but not as low as the center. The infraorbital triangle is sunk, the nasolabial furrow is smoothed out.",
            "The center of the upper lip is let down, the outer portions of the upper lip are pulled down but not as low as the center. The infraorbital triangle is lowered, the nasolabial furrow is softened.",
            "The center of the upper lip is depressed straight down, the outer portions of the upper lip are drawn down but not as low as the center. The infraorbital triangle is lowered, the nasolabial furrow is shallowed.",
            "The center of the upper lip is lowered directly, the outer portions of the upper lip are pulled down but not as low as the center. The infraorbital triangle is depressed, the nasolabial furrow is minimized.",
            "The center of the upper lip is drawn down, the outer portions of the upper lip are lowered but not as low as the center. The infraorbital triangle is depressed, the nasolabial furrow is lightened.",
            "The center of the upper lip is sunk down, the outer portions of the upper lip are dropped but not as low as the center. The infraorbital triangle is pulled down, the nasolabial furrow is softened.",
            "The center of the upper lip is released downwards, the outer portions of the upper lip are lowered but not as low as the center. The infraorbital triangle is depressed, the nasolabial furrow is reduced.",
            "The center of the upper lip is dropped down, the outer portions of the upper lip are pulled down but not as low as the center. The infraorbital triangle is lowered, the nasolabial furrow is minimized."],
        12: ["The corners of the lips are markedly lowered and angled down obliquely. The nasolabial furrow has softened slightly and is lowered obliquely slightly. The infraorbital triangle is lowered slightly.",
            "The corners of the lips are markedly dropped and angled down obliquely. The nasolabial furrow has lessened slightly and is depressed obliquely slightly. The infraorbital triangle is pulled down slightly.",
            "The corners of the lips are markedly depressed and angled down obliquely. The nasolabial furrow has smoothed out slightly and is lowered obliquely slightly. The infraorbital triangle is lowered slightly.",
            "The corners of the lips are markedly pulled down and angled down obliquely. The nasolabial furrow has shallowed slightly and is decreased obliquely slightly. The infraorbital triangle is depressed slightly.",
            "The corners of the lips are markedly flattened and angled down obliquely. The nasolabial furrow has reduced slightly and is lowered obliquely slightly. The infraorbital triangle is pulled down slightly.",
            "The corners of the lips are markedly lowered and angled down obliquely. The nasolabial furrow has diminished slightly and is lowered obliquely slightly. The infraorbital triangle is depressed slightly.",
            "The corners of the lips are markedly sagged and angled down obliquely. The nasolabial furrow has softened slightly and is lowered obliquely slightly. The infraorbital triangle is depressed slightly.",
            "The corners of the lips are markedly retracted and angled down obliquely. The nasolabial furrow has lessened slightly and is dropped obliquely slightly. The infraorbital triangle is lowered slightly.",
            "The corners of the lips are markedly relaxed and angled down obliquely. The nasolabial furrow has smoothed out slightly and is lowered obliquely slightly. The infraorbital triangle is pulled down slightly.",
            "The corners of the lips are markedly pushed down and angled down obliquely. The nasolabial furrow has reduced slightly and is lowered obliquely slightly. The infraorbital triangle is lowered slightly."],
        14: ["The lip corners are extremely relaxed, and the wrinkling as skin is released outwards around the lip corners is mild. The skin on the chin and lower lip is loosened away from the lip corners, and the lips are compressed and rounded away from the teeth.",
            "The lip corners are extremely softened, and the wrinkling as skin is expanded outwards around the lip corners is slight. The skin on the chin and lower lip is pulled back from the lip corners, and the lips are curled and protruded away from the teeth.",
            "The lip corners are extremely loosened, and the wrinkling as skin is released outwards around the lip corners is subtle. The skin on the chin and lower lip is withdrawn from the lip corners, and the lips are unfolded and rounded away from the teeth.",
            "The lip corners are extremely unconstrained, and the wrinkling as skin is pushed outwards around the lip corners is minimal. The skin on the chin and lower lip is drawn back from the lip corners, and the lips are rounded and relaxed away from the teeth.",
            "The lip corners are extremely slackened, and the wrinkling as skin is pulled outwards around the lip corners is light. The skin on the chin and lower lip is drawn away from the lip corners, and the lips are curved and softened away from the teeth.",
            "The lip corners are extremely weakened, and the wrinkling as skin is expanded outwards around the lip corners is gentle. The skin on the chin and lower lip is released away from the lip corners, and the lips are pushed back and relaxed against the teeth.",
            "The lip corners are extremely unfurled, and the wrinkling as skin is spread outwards around the lip corners is reduced. The skin on the chin and lower lip is drawn away from the lip corners, and the lips are softened and lifted away from the teeth.",
            "The lip corners are extremely released, and the wrinkling as skin is expanded outwards around the lip corners is light. The skin on the chin and lower lip is relaxed away from the lip corners, and the lips are folded and rounded away from the teeth.",
            "The lip corners are extremely loosened, and the wrinkling as skin is pushed outwards around the lip corners is mild. The skin on the chin and lower lip is pulled back from the lip corners, and the lips are rounded and distended away from the teeth.",
            "The lip corners are extremely unbound, and the wrinkling as skin is drawn outwards around the lip corners is slight. The skin on the chin and lower lip is relaxed away from the lip corners, and the lips are unfurled and softened against the teeth."],
        15: ["The lip corners are raised up slightly, with some medial relaxing and angling up of the corners, and slight smoothness and absence of bulges appear beyond the lip corners.",
            "The lip corners are elevated slightly, with some inward relaxing and angling up of the corners, and slight smoothing and absence of wrinkles appear beyond the lip corners.",
            "The lip corners are lifted slightly, with some inward relaxing and angling up of the corners, and slight tightness and softness appear beyond the lip corners.",
            "The lip corners are angled up slightly, with some loosening and angling up of the corners, and slight flatness and lack of bulges appear beyond the lip corners.",
            "The lip corners are drawn up slightly, with some inward relaxing and angling up of the corners, and slight smoothness and lack of wrinkles appear beyond the lip corners.",
            "The lip corners are lifted up slightly, with some inward relaxing and angling up of the corners, and slight firmness and absence of bulges appear beyond the lip corners.",
            "The lip corners are pulled back slightly, with some medial relaxing and angling up of the corners, and slight flatness and absence of wrinkles appear beyond the lip corners.",
            "The lip corners are pushed upward slightly, with some inward loosening and angling up of the corners, and slight smoothness and absence of bulges appear beyond the lip corners.",
            "The lip corners are raised up slightly, with some relaxing and angling up of the corners, and slight smoothing and lack of wrinkles appear beyond the lip corners.",
            "The lip corners are lifted slightly, with some inward relaxing and angling up of the corners, and slight softness and flatness appear beyond the lip corners."],
        17: ["The chin boss shows severe to extreme smoothness as it is pulled down severely, and the lower lip is pulled down and inward markedly.",
            "The chin boss shows severe to extreme relaxation as it is drawn down severely, and the lower lip is drawn down and back markedly.",
            "The chin boss shows severe to extreme softening as it is lowered severely, and the lower lip is lowered and retracted markedly.",
            "The chin boss shows severe to extreme firmness as it is pressed down severely, and the lower lip is pressed down and flattened markedly.",
            "The chin boss shows severe to extreme flattening as it is sunk down severely, and the lower lip is dropped down and back markedly.",
            "The chin boss shows severe to extreme easing as it is lowered severely, and the lower lip is dropped and pulled back markedly.",
            "The chin boss shows severe to extreme calmness as it is sunk down severely, and the lower lip is pulled back and down markedly.",
            "The chin boss shows severe to extreme relaxation as it is brought down severely, and the lower lip is retracted and pulled down markedly.",
            "The chin boss shows severe to extreme smoothing as it is drawn down severely, and the lower lip is flattened and brought back markedly.",
            "The chin boss shows severe to extreme soothing as it is lowered severely, and the lower lip is dropped down and retracted markedly."],
        23: ["The lips are loosened maximally and the red parts are widened maximally, creating extreme smoothness and flattening around the margins of the red parts of both lips.",
            "The lips are softened maximally and the red parts are expanded maximally, creating extreme smoothing and flattening around the margins of the red parts of both lips.",
            "The lips are relaxed maximally and the red parts are broadened maximally, creating extreme softness and smoothness around the margins of the red parts of both lips.",
            "The lips are opened maximally and the red parts are stretched maximally, creating extreme evenness and flattening around the margins of the red parts of both lips.",
            "The lips are released maximally and the red parts are expanded maximally, creating extreme flattening and softness around the margins of the red parts of both lips.",
            "The lips are unfurled maximally and the red parts are spread maximally, creating extreme calmness and smoothness around the margins of the red parts of both lips.",
            "The lips are unfurled maximally and the red parts are expanded maximally, creating extreme flatness and softness around the margins of the red parts of both lips.",
            "The lips are released maximally and the red parts are softened maximally, creating extreme smoothness and calmness around the margins of the red parts of both lips.",
            "The lips are flattened maximally and the red parts are spread maximally, creating extreme softening and smoothness around the margins of the red parts of both lips.",
            "The lips are dropped maximally and the red parts are relaxed maximally, creating extreme smoothness and calmness around the margins of the red parts of both lips."],
        24: ["The lips are loosely parted, relaxing skin above and below the red parts, with severe widening of the lips and smoothing above the upper lip.",
            "The lips are gently opened, reducing bulging skin above and below the red parts, with extreme broadening of the lips and flattening above the upper lip.",
            "The lips are slightly separated, softening skin above and below the red parts, with significant expansion of the lips and smoothing above the upper lip.",
            "The lips are released, reducing bulging skin above and below the red parts, with severe widening of the lips and softening above the upper lip.",
            "The lips are unpressed, easing skin above and below the red parts, with dramatic widening of the lips and smoothing above the upper lip.",
            "The lips are relaxed, minimizing bulging skin above and below the red parts, with major expansion of the lips and flattening above the upper lip.",
            "The lips are slackened, softening skin above and below the red parts, with extensive broadening of the lips and smoothing above the upper lip.",
            "The lips are spread apart, reducing skin bulging above and below the red parts, with noticeable widening of the lips and flattening above the upper lip.",
            "The lips are unpressed, loosening skin above and below the red parts, with extreme expansion of the lips and softening above the upper lip.",
            "The lips are opened widely, reducing bulging skin above and below the red parts, with significant widening of the lips and smoothing above the upper lip."],
        25: ["The teeth are barely concealed, and the lips are pressed together slightly. Nothing suggests that the jaw has raised even though the upper teeth are clearly visible.",
            "The teeth are not fully hidden, and the lips are sealed slightly. Nothing suggests that the jaw has tightened even though the upper teeth are clearly visible.",
            "The teeth are somewhat obscured, and the lips are tightly shut. Nothing suggests that the jaw has lifted even though the upper teeth are clearly visible.",
            "The teeth are partially covered, and the lips are firmly closed. Nothing suggests that the jaw has ascended even though the upper teeth are clearly visible.",
            "The teeth are not entirely visible, and the lips are drawn together slightly. Nothing suggests that the jaw has moved up even though the upper teeth are clearly visible.",
            "The teeth are somewhat hidden, and the lips are held tightly together. Nothing suggests that the jaw has compressed even though the upper teeth are clearly visible.",
            "The teeth are barely visible, and the lips are joined together slightly. Nothing suggests that the jaw has contracted even though the upper teeth are clearly visible.",
            "The teeth are not entirely hidden, and the lips are held together slightly. Nothing suggests that the jaw has tightened even though the upper teeth are clearly visible.",
            "The teeth are slightly covered, and the lips are pressed firmly together. Nothing suggests that the jaw has closed up even though the upper teeth are clearly visible.",
            "The teeth are not fully visible, and the lips are joined closely. Nothing suggests that the jaw has contracted even though the upper teeth are clearly visible."],
        26: ["The jaw is raised about as much as it can lift from tensing of the muscles. The lips are closed to about the extent that the jaw raising can produce.",
            "The jaw is elevated about as much as it can ascend from tightening of the muscles. The lips are sealed to about the extent that the jaw elevation can produce.",
            "The jaw is lifted about as much as it can rise from contracting of the muscles. The lips are together to about the extent that the jaw lifting can produce.",
            "The jaw is pulled up about as much as it can ascend from stiffening of the muscles. The lips are joined to about the extent that the jaw pulling up can produce.",
            "The jaw is brought up about as much as it can ascend from tightening of the muscles. The lips are held together to about the extent that the jaw bringing up can produce.",
            "The jaw is lifted about as much as it can rise from firming of the muscles. The lips are shut to about the extent that the jaw lifting can produce.",
            "The jaw is elevated about as much as it can lift from contracting of the muscles. The lips are closed tightly to about the extent that the jaw elevation can produce.",
            "The jaw is raised about as much as it can lift from tensing of the muscles. The lips are drawn together to about the extent that the jaw raising can produce.",
            "The jaw is held up about as much as it can rise from stiffening of the muscles. The lips are pursed to about the extent that the jaw holding up can produce.",
            "The jaw is drawn up about as much as it can lift from tightening of the muscles. The lips are compressed to about the extent that the jaw drawing up can produce."]
    }
    return [au_descriptions.get(au, "No description available for this AU.") for au in aus]

if __name__ == '__main__':
    au_des_list = describe_au_all()
    for k, v in au_des_list.items():
        print(f'{k}: {v}')