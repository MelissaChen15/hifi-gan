# log for ACC 2022

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
