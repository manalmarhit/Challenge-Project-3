import numpy as np
import matplotlib.pyplot as plt

# All 227 makemore generated names
generated_names = [
    "maddoxxz","habab","schning","farrena","raig","ellizavder","zanyam","obaj",
    "adanson","dsaia","ziyaan","galirah","ahla","tallis","khaizagr","jessin",
    "lashela","alei","imrea","youx","saylynn","amiana","emou","mirajul","priton",
    "yannisset","emino","kenlingta","skayley","delcin","nikka","joh","kynlamie",
    "nako","vieram","balli","karmelyn","jasa","haycin","anison","malauna","venacre",
    "osifata","emmiliane","allina","castan","ismer","ineomer","juliansh","patrish",
    "shorin","malcaten","margarettero","azbal","makaeena","perssie","rafniel","shady",
    "aristez","dilka","betzamy","haedy","gynezmae","rognad","markeyah","renedicia",
    "nadilyn","mallis","daryck","alki","liddicley","ondi","wnytten","danim","fesson",
    "olywaaf","mouel","daranika","azron","karish","anando","iolan","fer","aliara",
    "gerella","damear","alinshku","melasia","wallow","quontree","daleyah","lumy",
    "racey","misosia","khailan","leddyn","elane","hechodaida","joyc","zakiiah","klyna",
    "sannak","jellee","mawking","adesa","miti","kaptia","yevan","hayde","nillie",
    "lewesty","corio","johanobi","alylian","kynsendy","deatrya","emary","malieka",
    "kayyan","ramariah","jahirson","peagen","joesh","pravone","firkmy","breatt",
    "josselde","abubakum","annaz","cvarlin","ravneck","alnise","treniti","wayling",
    "praire","laowoluwan","jenidiss","khylo","dekk","chiocn","montaliose","tabastiun",
    "julynl","marilus","sakyo","aricabhire","aoralee","wesams","adolwamiz","saflyn",
    "rayyam","bobia","marliee","kencent","umai","holf","trastlant","azleigh","granvir",
    "anleigh","orat","shilviv","osren","kolf","kydrah","kacieon","dal","rubenna",
    "zahla","mickeemion","breylynn","tykio","benkie","eferson","surayya","kittaly",
    "yacez","kualar","hroen","katdel","vishormee","selaniy","asna","daxtan",
    "aslmet","wohaiti","chebi","xania","hannald","lufe","azvin","royan","eilannes",
    "ayandrea","layleigh","dalea","phontae","nyami","higa","dedez","caroland",
    "kahlida","deankai","lakende","anjela","kollett","quadry","beurton","edyin",
    "yarcielos","mahikatrif","anwelm","xzura","aadrianna","azelyjah","alymiah",
    "shavyn","jezrion","adnae","iraan","alond","elliafoluwa","aadee","yaryus","elien"
]

# Setup
chars = ['.'] + list('abcdefghijklmnopqrstuvwxyz')
n = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Build transition matrix
matrix = np.zeros((n, n), dtype=int)
for name in generated_names:
    word = '.' + name + '.'
    for ch1, ch2 in zip(word, word[1:]):
        if ch2 in char_to_idx:  # skip any unexpected characters
            matrix[char_to_idx[ch1]][char_to_idx[ch2]] += 1

# Convert to probabilities
prob_matrix = matrix / matrix.sum(axis=1, keepdims=True)

# Plot heatmap
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(prob_matrix, cmap='Blues')

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(chars)
ax.set_yticklabels(chars)

ax.set_xlabel("Second Letter")
ax.set_ylabel("First Letter")
ax.set_title("Bigram Transition Probabilities (Makemore Gnerated Names)")

plt.colorbar(im)
plt.tight_layout()
plt.savefig('heatmap_makemore.png', dpi=150)
print("Saved heatmap_makemore.png")