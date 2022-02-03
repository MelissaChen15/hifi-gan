# log 

[!]first attemp: using the pre-trained univeral discriminator to do inference
    - [x]print out the loss and observe the pattern
    using asvsppof2019la eval set, #utt=71113
    the asvspoof2019la is 16k, but our model is 22k -> leave it
    for i in a:
	splt  = str(i[0]).split("/")
	if "spoof" in splt[7]:
		spoof.append(i)
	else:
		bona.append(i)
    - result: cannot distinguish from loss_disc_s, losses_disc_f using simple nn


pd.DataFrame(spoof).to_csv("spoof.csv")
pd.DataFrame(bona).to_csv("bona.csv")

    - []view current discriminators as feature extractors, add classifier
        using last layer output as features
        using rawnet as classifier, dim is hard-coded 

01/02/2022 - 18:15
pretrained 16k model -> test success
replicating exps from last part
need to use train set to train!
ls mels_hifigan/ -1 | sed -e 's/\.npy$//'>> eval.txt

find out the dim of the extracted feature map

02/02/2022 - 15:29

check:
from extract.py
spoof_LA_E_1161920.wav [0.28896820545196533, 0.2512284517288208, 0.26991185545921326, 0.29238131642341614, 0.2133161723613739] [0.2313181757926941, 1.5503935813903809, 0.48195528984069824]
from models.py
[0.30386558175086975, 0.17165113985538483, 0.24869920313358307, 0.3421728312969208, 0.3009723126888275, 0.18500253558158875, 1.2594478130340576, 0.42712709307670593, 0, 'spoof_LA_E_1161920']
it is not the same segment!
训练的时候是用一个segment的mel，inference的时候是用整个utt的mel -> 之后可以改用LFCC训练hifigan

(b, 1, samples) torch.Size([16, 1, 8192])
# df : (b, x, x, x) ds: (b, x, x)