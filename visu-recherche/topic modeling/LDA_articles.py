import re
from collections import defaultdict, Counter
import pandas as pd
import prince  # pour l’Analyse en Correspondance
from keybert import KeyBERT
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from prince import CA
from gensim.models.coherencemodel import CoherenceModel
from Dictionnaire_halids_keywords import load_keywords
import unicodedata
# --- Configuration ---
HAL_LIST = ['hal-04762097', 'hal-04736454', 'hal-04768296', 'hal-04665063', 'hal-04764247', 'hal-04695595', 'hal-04701759', 'hal-04720291', 'hal-04685184', 'hal-04801861', 'hal-04632526', 'hal-04617131', 'hal-04640068', 'hal-04705811', 'hal-04645968', 'hal-04614241', 'hal-04574640', 'hal-04616517', 'hal-04593399', 'hal-04629995', 'hal-04631163', 'hal-04794145', 'hal-04630089', 'halshs-04654217', 'hal-04602229', 'hal-04001898', 'hal-04428828', 'hal-04541350', 'hal-04544157', 'hal-04358467', 'hal-04424100', 'hal-04360221', 'hal-04539329', 'hal-04479188', 'hal-04423979', 'hal-04419041', 'hal-04356813', 'hal-04432659', 'hal-04423348', 'halshs-04579125', 'hal-04729913', 'hal-04036482', 'hal-04593961', 'hal-04577420', 'hal-04655069', 'hal-04688068', 'hal-04593480', 'hal-04548715', 'hal-04578273', 'hal-04427829', 'hal-04539879', 'hal-04593465', 'hal-04254949', 'hal-03442137', 'hal-03615137', 'hal-04611461', 'hal-04485065', 'hal-04242023', 'hal-04390768', 'hal-04593403', 'hal-04574946', 'hal-04644398', 'hal-04216055', 'hal-04155178', 'hal-04202766', 'hal-04260042', 'hal-04160013', 'hal-04186048', 'hal-04574970', 'hal-04265346', 'hal-04172863', 'hal-04251755', 'hal-04206447', 'hal-04254405', 'hal-04321188', 'hal-04440353', 'hal-04212792', 'hal-03832480', 'hal-04135264', 'hal-04216175', 'hal-04216177', 'hal-04253761', 'hal-04143015', 'hal-04254122', 'hal-04593478', 'hal-04273545', 'hal-04131585', 'hal-04130213', 'hal-04130132', 'hal-04093374', 'hal-04029145', 'hal-04048829', 'hal-04076307', 'hal-04181570', 'hal-04087420', 'hal-03959766', 'ujm-04165556', 'hal-04213215', 'hal-04126067', 'hal-04087415', 'hal-04250963', 'hal-04075832', 'hal-04205024', 'hal-04319492', 'hal-03920536', 'hal-03670085', 'hal-04258472', 'hal-04267977', 'hal-04001887', 'hal-02318267', 'hal-04253742', 'hal-04254953', 'hal-04112575', 'hal-04168456', 'hal-04268013', 'hal-04253752', 'hal-04276030', 'hal-04038023', 'hal-04254778', 'hal-03913356', 'hal-04590548', 'hal-04244852', 'hal-03228252', 'hal-03164338', 'hal-04276738', 'hal-04575126', 'hal-03990543', 'hal-03860468', 'hal-04044494', 'hal-03601330', 'hal-03767631', 'hal-04182653', 'hal-02362067', 'hal-03794948', 'hal-03581457', 'hal-02564349', 'hal-03559365', 'hal-04268018', 'hal-03821125', 'hal-02493319', 'hal-03657196', 'hal-03295581', 'hal-04277391', 'hal-04044566', 'hal-03562371', 'hal-04276085', 'hal-03860460', 'hal-03888760', 'hal-04573531', 'hal-04590498', 'hal-03959759', 'hal-03870605', 'hal-04494204', 'hal-03870592', 'hal-03780032', 'hal-03860497', 'hal-03903647', 'hal-03830604', 'hal-03680792', 'hal-03787203', 'hal-03782827', 'hal-03671852', 'hal-03671851', 'hal-04275993', 'hal-04297354', 'hal-03801053', 'hal-03860830', 'hal-03845277', 'hal-03773368', 'hal-03817736', 'hal-03759651', 'hal-03759647', 'hal-03787205', 'hal-04268476', 'hal-03807108', 'hal-03670855', 'hal-03670586', 'hal-03670577', 'hal-03759597', 'hal-04253784', 'hal-03860827', 'hal-04273536', 'hal-04276012', 'hal-04166172', 'hal-03637425', 'hal-03708610', 'hal-03935833', 'hal-03990509', 'hal-04540314', 'hal-03559402', 'hal-03848222', 'hal-03848224', 'hal-03559398', 'hal-03537148', 'hal-03860881', 'hal-04575362', 'hal-03602455', 'hal-03727169', 'hal-03701451', 'hal-03821095', 'hal-04168435', 'hal-04255228', 'hal-04253771', 'hal-03727181', 'hal-04083262', 'hal-03494781', 'hal-03109686', 'hal-03413484', 'hal-03577262', 'hal-03356164', 'hal-03330800', 'hal-03349734', 'hal-03349492', 'hal-03423971', 'hal-03574595', 'hal-03344680', 'hal-03574609', 'hal-04087296', 'hal-03329932', 'hal-03616634', 'hal-03409892', 'hal-03409889', 'hal-03562701', 'hal-03349673', 'hal-03219350', 'hal-03298695', 'hal-03559362', 'hal-03559364', 'hal-03344668', 'hal-03413460', 'hal-03470367', 'hal-04087246', 'hal-03426962', 'hal-04276017', 'hal-03345735', 'hal-03255349', 'hal-04277423', 'hal-03601265', 'hal-03255341', 'hal-03259801', 'hal-03167498', 'hal-03259013', 'hal-03979752', 'hal-03563675', 'hal-03559370', 'hal-03256451', 'hal-03423979', 'hal-03208323', 'hal-03224929', 'hal-04087328', 'hal-03353642', 'hal-02978978', 'hal-03132940', 'hal-02985794', 'hal-03559386', 'hal-03132984', 'hal-02984117', 'hal-03562162', 'hal-03122020', 'hal-03100014', 'hal-03073936', 'hal-03265871', 'hal-03428910', 'hal-03409678', 'hal-03190532', 'hal-02429681', 'hal-03310455', 'hal-03341560', 'hal-04044428', 'hal-01383554', 'hal-03147824', 'hal-03189235', 'hal-03269127', 'hal-02088860', 'hal-02493338', 'hal-04044542', 'hal-03188029', 'hal-02614605', 'hal-02933051', 'hal-03255319', 'hal-04587615', 'hal-02985867', 'hal-03255334', 'hal-03428883', 'hal-02366280', 'hal-04590157', 'hal-03123038', 'hal-03235295', 'hal-04297374', 'hal-02366337', 'hal-03134854', 'hal-03134851', 'hal-03187452', 'hal-03152058', 'hal-04087260', 'hal-02910344', 'hal-03127155', 'hal-03200161', 'hal-02934433', 'hal-04589946', 'hal-02996940', 'hal-04277560', 'hal-02932485', 'hal-02507316', 'hal-03233337', 'hal-03132996', 'hal-03126870', 'hal-03126861', 'hal-02932836', 'hal-02899036', 'hal-02927208', 'hal-02547012', 'hal-02934517', 'hal-02933475', 'hal-02933466', 'hal-02933487', 'hal-02933469', 'hal-02933476', 'hal-02481374', 'hal-02477242', 'hal-02713204', 'hal-02713178', 'hal-02457063', 'hal-02389159', 'hal-02457075', 'hal-02747449', 'hal-02456651', 'hal-02942182', 'hal-04448257', 'hal-02456643', 'hal-02466289', 'hal-02914840', 'hal-03126876', 'hal-03269112', 'hal-03559387', 'hal-03269116', 'hal-02873020', 'hal-03269119', 'hal-03269118', 'hal-03269125', 'hal-02953469', 'hal-03162808', 'hal-03270164', 'hal-01613583', 'hal-04182657', 'hal-02873600', 'hal-03121830', 'hal-02924471', 'hal-02369882', 'hal-03985545', 'hal-02185060', 'hal-04276039', 'hal-02615187', 'hal-04275915', 'hal-03134847', 'hal-02878302', 'hal-02506409', 'hal-04081621', 'hal-01440269', 'hal-03189232', 'hal-02433213', 'hal-02923548', 'hal-03152084', 'hal-02077745', 'hal-02399993', 'hal-01709825', 'hal-03148343', 'hal-02307248', 'hal-03148361', 'hal-02895173', 'hal-02463908', 'hal-01941152', 'hal-02121090', 'hal-01958485', 'hal-01502252', 'hal-02269132', 'hal-02365397', 'hal-02055682', 'hal-02369439', 'hal-02068670', 'hal-01497104', 'hal-02288041', 'hal-02288043', 'hal-02288044', 'hal-02287983', 'hal-01725134', 'hal-03187410', 'hal-02618085', 'hal-02078108', 'hal-02280948', 'hal-02280944', 'hal-02365297', 'hal-02365285', 'hal-02007612', 'hal-02372376', 'hal-02369435', 'hal-02420416', 'hal-02448917', 'hal-02457735', 'hal-02419361', 'hal-02457638', 'hal-02366954', 'hal-02371140', 'hal-02943462', 'hal-02912374', 'hal-04277461', 'hal-02280472', 'hal-02291896', 'hal-02380780', 'hal-02437207', 'hal-02495516', 'hal-03175885', 'hal-02166428', 'hal-02291882', 'hal-02461801', 'hal-03152031', 'hal-02370820', 'hal-02457184', 'hal-02288063', 'hal-02367908', 'hal-02288565', 'hal-02288067', 'hal-02381367', 'hal-04289665', 'hal-02269139', 'hal-02382428', 'hal-03269089', 'hal-02365327', 'hal-02463910', 'hal-03269101', 'hal-02346147', 'hal-02365318', 'hal-02420403', 'hal-02005106', 'hal-02372076', 'hal-02461824', 'hal-02288519', 'hal-02288552', 'hal-02051399', 'hal-02370842', 'hal-02457728', 'hal-01900037', 'hal-02007601', 'hal-04267912', 'hal-02007623', 'hal-02371075', 'hal-02094838', 'hal-02287991', 'hal-02287988', 'hal-02292437', 'hal-04276038', 'hal-02019103', 'hal-01795319', 'hal-04589634', 'hal-02422892', 'hal-02943467', 'hal-02912385', 'hal-01833398', 'hal-01922988', 'hal-02288518', 'hal-01950907', 'hal-01810775', 'lirmm-01766795', 'hal-02292460', 'hal-02287759', 'hal-02912471', 'hal-02288509', 'hal-04267897', 'hal-02369904', 'hal-01724272', 'hal-02943469', 'hal-01714909', 'hal-01812011', 'hal-02287580', 'hal-02287764', 'hal-02287945', 'hal-02287919', 'hal-04277497', 'hal-02288000', 'hal-02288522', 'hal-02287951', 'hal-01815255', 'hal-02713307', 'hal-01652152', 'inserm-01847873', 'hal-02288042', 'hal-02287871', 'hal-02881802', 'hal-02287962', 'hal-02371087', 'hal-04277535', 'hal-01970744', 'hal-02382638', 'hal-01718718', 'hal-01584755', 'hal-02287949', 'hal-04587369', 'hal-01797151', 'hal-01779074', 'hal-02287837', 'hal-02287831', 'hal-02287832', 'hal-02287842', 'hal-01721650', 'hal-02287766', 'hal-02287765', 'hal-02190639', 'hal-02287753', 'hal-01893410', 'hal-02705056', 'hal-02023057', 'hal-02107531', 'hal-02287465', 'hal-01269137', 'hal-01360647', 'hal-01682750', 'hal-01975523', 'hal-02023085', 'hal-01679078', 'hal-03189231', 'hal-02287734', 'hal-02287879', 'hal-02107523', 'hal-01588129', 'hal-01745692', 'hal-02287610', 'hal-02287363', 'hal-01362864', 'hal-01537200', 'hal-04087237', 'hal-01361434', 'hal-02287001', 'hal-01370542', 'hal-01544680', 'hal-02287614', 'hal-01333295', 'hal-02704911', 'hal-01404966', 'hal-02287581', 'hal-02288504', 'hal-02287629', 'hal-02288505', 'hal-01636627', 'hal-01593459', 'hal-01662421', 'hal-02943475', 'hal-02365695', 'hal-01548475', 'hal-01548488', 'hal-01548469', 'hal-01548508', 'hal-02287731', 'hal-02287698', 'hal-01576857', 'hal-01722902', 'hal-01722906', 'hal-01840082', 'hal-01540479', 'hal-01540484', 'hal-01540481', 'hal-01725141', 'hal-02912377', 'hal-02287621', 'hal-02288492', 'hal-02287918', 'hal-02287481', 'hal-01580091', 'hal-01531243', 'hal-01531238', 'hal-02365713', 'hal-01531259', 'hal-02422947', 'hal-01531252', 'hal-01577813', 'hal-01618447', 'hal-02287881', 'hal-02287607', 'hal-02288511', 'hal-01567617', 'hal-01412059', 'hal-02912472', 'hal-02288463', 'hal-02288535', 'hal-02287480', 'hal-02287914', 'hal-02287913', 'hal-02287915', 'hal-02287829', 'hal-02287800', 'hal-02287799', 'hal-02287763', 'hal-01519728', 'hal-03991123', 'hal-02395677', 'hal-02912384', 'hal-01484744', 'hal-01447977', 'hal-01416357', 'hal-01416347', 'hal-02713341', 'hal-01438851', 'hal-01416366', 'hal-01416355', 'hal-01401988', 'hal-01400965', 'hal-02288528', 'hal-02287880', 'hal-02287762', 'hal-01723250', 'hal-01538184', 'hal-01538185', 'hal-01497087', 'hal-01354064', 'hal-01367546', 'hal-01272327', 'hal-02287361', 'hal-01393959', 'hal-02288472', 'hal-01337860', 'hal-01393964', 'hal-02412199', 'hal-02287866', 'hal-02943480', 'hal-02287356', 'hal-02287268', 'hal-02287401', 'hal-02287455', 'hal-01322937', 'hal-01419050', 'hal-01316485', 'hal-01353252', 'hal-02288525', 'hal-02412201', 'hal-02288480', 'hal-02412202', 'hal-02287387', 'hal-02287368', 'hal-02287369', 'hal-01418963', 'hal-01347167', 'hal-01306605', 'hal-01313567', 'cea-01843181', 'hal-01306596', 'hal-01329315', 'hal-02165428', 'hal-02287453', 'hal-02287370', 'hal-02287434', 'hal-01589331', 'hal-02412204', 'hal-01248013', 'hal-02287313', 'hal-02287267', 'hal-02287863', 'hal-01393968', 'hal-02288453', 'hal-02640960', 'hal-01248010', 'hal-02287438', 'hal-02288523', 'hal-02287867', 'hal-01248011', 'hal-01248014', 'hal-01490135', 'hal-01248012', 'hal-03181846', 'hal-02287020', 'hal-02107516', 'hal-03189229', 'hal-01237226', 'hal-01370051', 'hal-01374313', 'hal-03189227', 'hal-02287098', 'hal-02107503', 'hal-01354187', 'hal-01153882', 'hal-01259850', 'hal-01260588', 'hal-01120685', 'hal-02287222', 'hal-02286968', 'hal-02287099', 'hal-01087687', 'hal-01359236', 'hal-02637720', 'hal-01080955', 'hal-01601982', 'hal-01413791', 'hal-01567869', 'hal-01327662', 'hal-01321471', 'hal-01183959', 'hal-02107496', 'hal-02287205', 'hal-00936316', 'hal-02287265', 'hal-01190028', 'hal-01428019', 'hal-01354192', 'hal-01194354', 'hal-03179752', 'hal-01078073', 'hal-02704820', 'hal-02287317', 'hal-01030799', 'hal-01707390', 'hal-02286817', 'hal-01353855', 'hal-01197650', 'hal-01845537', 'hal-01164121', 'hal-00458648', 'hal-00984064', 'hal-00854458', 'hal-01468900', 'hal-00755255', 'hal-01058447', 'hal-01214665', 'hal-01214667', 'hal-02287197', 'hal-02287282', 'hal-01170924', 'hal-01219635', 'hal-01219637', 'hal-01890049', 'hal-02412178', 'hal-02287206', 'hal-01199648', 'hal-02437193', 'hal-01206808', 'hal-02912383', 'hal-02287216', 'hal-02287266', 'hal-02943532', 'hal-01199631', 'hal-01179556', 'cea-01841169', 'hal-01430858', 'hal-01259936', 'hal-01259917', 'hal-01206804', 'hal-01187061', 'hal-01270543', 'hal-01272370', 'hal-02287864', 'hal-01259923', 'hal-02287865', 'hal-01123542', 'hal-02412122', 'hal-02287868', 'hal-02412126', 'hal-01110035', 'hal-01110028', 'hal-02713516', 'hal-01110032', 'hal-02286986', 'hal-02288433', 'hal-02287003', 'hal-02287014', 'hal-02412124', 'hal-02412189', 'hal-02287882', 'hal-01098143']






INPUT_FILE = "../pdf_scraping/articles_de_publications_permanents_formatted.txt"
TARGET_AUTHORS = {
        "Roland Badeau", "Pascal Bianchi", "Philippe Ciblat",
        "Stephan Clémençon", "Florence d'Alché-Buc", "Slim Essid",
        "Olivier Fercoq", "Pavlo Mozharovskyi", "Geoffroy Peeters",
        "Gaël Richard", "François Roueff", "Maria Boritchev",
        "Radu Dragomir", "Mathieu Fontaine", "Ekhiñe Irurozki",
        "Yann Issartel", "Hicham Janati", "Ons Jelassi",
        "Matthieu Labeau", "Charlotte Laclau",
        "Laurence Likforman-Sulem", "Yves Grenier"
    }
NUM_TOPICS = 10  # ajustez ce nombre selon vos besoins
def canonique(nom):
    """
    Supprime les espaces, accents, met en minuscules.
    """
    nom = nom.replace(" ", "").lower()
    nom = unicodedata.normalize('NFD', nom)
    nom = ''.join(c for c in nom if unicodedata.category(c) != 'Mn')
    return nom

def nom_est_dans_liste(nom, liste):
    """
    Vérifie si le nom (normalisé) est présent dans la liste (normalisée).
    """
    nom_can = canonique(nom)
    liste_can = [canonique(n) for n in liste]
    return nom_can in liste_can
def filter_articles_by_topic(lda_model, corpus_bow, articles_dict, doc2hal_id, topic_idx, threshold=0.7):
    """
    Filtrer les articles associés à un topic donné si leur probabilité dépasse un seuil.
    """
    filtered_articles = {}
    for doc_idx, bow in enumerate(corpus_bow):
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0)
        prob = dict(doc_topics).get(topic_idx, 0)
        if prob >= threshold:
            hal_id = doc2hal_id[doc_idx]
            if hal_id in articles_dict:
                filtered_articles[hal_id] = articles_dict[hal_id].copy()
                filtered_articles[hal_id]['topic_prob'] = prob
    return filtered_articles

def concat_title_keywords(filtered_articles):
    """
    Prend un dict d'articles (comme filtered) et retourne un dict
    avec hal_id comme clé et la concaténation 'title + keywords' en texte.
    """
    concatenated = {}
    for hal_id, info in filtered_articles.items():
        title = info.get('title', '')
        keywords = info.get('keywords', [])
        keywords_text = ' '.join(keywords) if keywords else ''
        concatenated_text = f"{title} {keywords_text}".strip()
        concatenated[hal_id] = concatenated_text
    return concatenated

from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams(texts_dict, n=2, min_freq=1):
    """
    Extrait les n-grammes les plus fréquents à partir d'un dict de textes.
    """
    documents = list(texts_dict.values())
    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=1 , max_df = 1000)
    X = vectorizer.fit_transform(documents)
    freqs = X.sum(axis=0).A1
    ngrams = vectorizer.get_feature_names_out()
    ngram_freq = dict(zip(ngrams, freqs))
    ngram_freq = dict(sorted(ngram_freq.items(), key=lambda item: item[1], reverse=True))
    return ngram_freq

def filter_ngrams_by_lda_keywords(lda_model, dictionary, topic_id, ngram_freq, topn_keywords=20):
    """
    Filtre et ordonne les n-grammes selon la présence des top mots clés LDA.
    """
    topic_terms = lda_model.get_topic_terms(topicid=topic_id, topn=topn_keywords)
    topic_keywords = [dictionary[id_word] for id_word, prob in topic_terms]
    filtered_ngrams = []
    seen_ngrams = set()
    for kw in topic_keywords:
        for ngram, freq in ngram_freq.items():
            if kw.lower() in ngram.lower() and ngram not in seen_ngrams:
                filtered_ngrams.append((ngram, freq))
                seen_ngrams.add(ngram)
    return filtered_ngrams

def main():
    

    # --- 1. Lire et découper le fichier en sections "Thèse N : ..." ---
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        contenu = f.read()

    # Trouver les positions de chaque "Thèse N : titre"
    header_re = re.compile(r'^Thèse\s*(\d+)\s*:\s*(.*?)\s*$', re.MULTILINE)
    matches = list(header_re.finditer(contenu))

    documents = []
    doc_authors = []

    for idx, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(contenu)
        section = contenu[start:end].strip()

        # Séparer le bloc "auteurs" (jusqu'à la première ligne vide) du reste du texte
        try:
            authors_block, body = section.split('\n\n', 1)
        except ValueError:
            authors_block, body = section, ""

        # Nettoyage de la liste d'auteurs
        authors = []
        for part in re.split(r',|\n', authors_block):
            name = part.strip().rstrip(',')
            if name in TARGET_AUTHORS:
                authors.append(name)

        # Conserver seulement les sections avec au moins un auteur cible
        if not authors:
            continue

        # Préparer le texte du document (titre + corps)
        full_text = title + " " + body.replace('\n', ' ')
        tokens = [tok for tok in simple_preprocess(full_text) if tok not in STOPWORDS]

        documents.append(tokens)
        doc_authors.append(authors)

    # --- 2. Construire le dictionnaire et le corpus pour LDA ---
    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # --- 3. Entraîner le modèle LDA ---
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=42,
        passes=10,
        alpha="auto",
        per_word_topics=False
    )

    # --- 4. Attribuer à chaque document son topic dominant ---

    doc_topics = []
    for bow in corpus:
        topics_probs = lda.get_document_topics(bow)
        dominant_topic = max(topics_probs, key=lambda x: x[1])[0]
        doc_topics.append(dominant_topic)

    topic_doc = [[] for _ in range(NUM_TOPICS)]
    for i, topic in enumerate(doc_topics):
        topic_doc[topic].append(i)


    # --- Afficher les mots-clés associés à chaque topic ---
    print("\nMots-clés par topic :")
    for topic_id in range(NUM_TOPICS):
        print(f"Topic {topic_id} : ", end='')
        words = lda.show_topic(topic_id, topn=30)  # topn=30 mots par topic
        keywords = ", ".join([word for word, prob in words])
        print(keywords)

    # 1. Score de cohérence c_v
    coherence_model = CoherenceModel(
        model=lda,
        texts=documents,
        dictionary=dictionary,
        coherence='c_v'
    )
    print("Score de cohérence (c_v) :", coherence_model.get_coherence())

    return {
        "lda": lda,
        "corpus": corpus,
        "dictionary": dictionary,
        "HAL_LIST": HAL_LIST,
        "documents": documents
    }

def etiquettage(number_of_topics):
    noms_pris = []
    noms=["" for i in range(number_of_topics)]
    for n in range(number_of_topics) :
          
             List =bigrams_freq[n] 
             print(List)
             if List:
                max_key = max(List.keys(), key=lambda x: List[x])
                while max_key[0] in noms_pris :
                    max_key = max(List.keys(), key=lambda x: List[x])
                    List.pop(max_key, None)
                    if List :
                          max_key = max(List.keys(), key=lambda x: List[x])
                    else:
                        noms[n]=" "
                        break
                noms[n]=max_key
                noms_pris.append(max_key)
             else: noms[n]=" "
    print (noms)
    return noms
if __name__ == "__main__":


    # Exécution du pipeline principal
    results = main()
    lda = results["lda"]
    corpus = results["corpus"]
    dictionary = results["dictionary"]
    HAL_LIST = results["HAL_LIST"]
    DIC_HALIDS = load_keywords()
    print(dictionary)
    # 1. Filtrage des articles par topic
    filtered =[ filter_articles_by_topic(
        lda, corpus, DIC_HALIDS, {i: hal_id for i, hal_id in enumerate(HAL_LIST)}, topic_idx, threshold=0.1) for topic_idx in range(NUM_TOPICS) ]
    
    for i in range(NUM_TOPICS):
        print(f"\nTopic {i} : {len(filtered[i])} articles")
        for hal_id, info in filtered[i].items():
            print(f"{hal_id}: {info['title']} (probabilité: {info['topic_prob']:.4f})")
       

    # 2. Concaténation titre + keywords
    concatenated_texts = [concat_title_keywords(filtered[i]) for i in range(NUM_TOPICS)]
  

    # 3. Extraction des bigrammes
    bigrams_freq = [extract_ngrams(text, n=2, min_freq=2) for text in concatenated_texts]
    for i, bigrams in enumerate(bigrams_freq):
        print(f"\nTop bigrams for topic {i}:")
        for ngram, freq in list(bigrams.items())[:10]:
            print(f"{ngram}: {freq}")
  

    # 4. Filtrage des n-grammes par mots-clés LDA
    filtered_ngrams = [filter_ngrams_by_lda_keywords(lda, dictionary, topic_id=i, ngram_freq=bigrams, topn_keywords=20) for i, bigrams in enumerate(bigrams_freq)]

    for i, filtered in enumerate(filtered_ngrams):
        print(f"\nFiltered n-grams for topic {i}:")
        for ngram, freq in filtered[:10]:
            print(f"{ngram}: {freq}")

    
    author_topic_contrib = defaultdict(lambda: defaultdict(float))
    filtered =[ filter_articles_by_topic(
        lda, corpus, DIC_HALIDS, {i: hal_id for i, hal_id in enumerate(HAL_LIST)}, topic_idx, threshold=0.1) for topic_idx in range(NUM_TOPICS) ]
    
    for  articles in filtered:
        for article_info in articles.values():
            authors = article_info.get("authors", [])
            prob = article_info.get("topic_prob", 0)
            for author in authors:
                author_topic_contrib[author][filtered.index(articles)] += prob

    author_topic_percent = {}
    for author, topic_scores in author_topic_contrib.items():
        total = sum(topic_scores.values())
        if total > 0:
            author_topic_percent[author] = {
                topic: round(score / total * 100, 2) for topic, score in topic_scores.items()
            }

    # Affichage
    for author, topics in author_topic_percent.items():
        print(f"\nAuteur: {author}")
        for topic_id, percent in sorted(topics.items()):
            print(f"  Topic {topic_id} : {percent}%")
    

    # 1. Construction de la matrice Auteurs x Topics
    # 1. Appliquer la fonction canonique sur les noms d'auteurs
    author_topic_percent_canonique = defaultdict(lambda: defaultdict(float))
    canonical_to_real_name = {}

    # Itérer sur tous les auteurs et agrégations de leurs contributions par leur nom canonique
    for author, topics in author_topic_percent.items():
        canonical_name = canonique(author)  # Appliquez votre fonction canonique sur le nom de l'auteur
        
        # Si nous n'avons pas encore enregistré de nom réel pour ce nom canonique, enregistrons-le
        if canonical_name not in canonical_to_real_name:
            canonical_to_real_name[canonical_name] = author
        
        # Agrégation des contributions des auteurs
        for topic, percent in topics.items():
            author_topic_percent_canonique[canonical_name][topic] += percent

    # 2. Remplacer le nom canonique par le nom réel dans le dictionnaire agrégé
    author_topic_percent_real_name = {
        canonical_to_real_name[canonical_name]: topics
        for canonical_name, topics in author_topic_percent_canonique.items()
    }

    # 3. Création du DataFrame en utilisant les noms réels
    df_matrix = pd.DataFrame(author_topic_percent_real_name).T.fillna(0)

    # Nettoyage pour éviter les problèmes avec prince.CA
    df_matrix_clean = df_matrix.copy()

    # S'assurer que toutes les colonnes sont numériques
    df_matrix_clean = df_matrix_clean.apply(pd.to_numeric, errors='coerce')

    # Remplir les valeurs manquantes
    df_matrix_clean = df_matrix_clean.fillna(0)

    # 4. Application de l'Analyse en Correspondance
    ca = CA(n_components=2, n_iter=100, copy=True, check_input=True, engine='sklearn')

    # Exécution de l'Analyse en Correspondance
    ca = ca.fit(df_matrix_clean)

    # Récupération des coordonnées pour les auteurs et les topics
    row_coords = ca.row_coordinates(df_matrix_clean)      # Auteurs
    col_coords = ca.column_coordinates(df_matrix_clean)   # Topics
    print(df_matrix.columns)
    print(col_coords)
    # 2. Fusionner toutes les coordonnées
    all_x = list(row_coords[0]) + list(col_coords[0])
    all_y = list(row_coords[1]) + list(col_coords[1])

    # 3. Calcul des bornes avec zoom serré
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # 4. Calcul d'une petite marge (5%)
    padding_x = (max_x - min_x) * 0.05
    padding_y = (max_y - min_y) * 0.05

    # 5. Création de la figure
    fig, ax = plt.subplots(figsize=(1, 2))

    # 6. Scatter pour voir les points
    #ax.scatter(row_coords[0], row_coords[1], color='blue', s=20, alpha=0.6, label="Auteurs")
    #ax.scatter(col_coords[0], col_coords[1], color='red', s=40, marker='X', alpha=0.8, label="Topics")

    L=[]
    # Tracer les auteurs
    """ 
    
    for i, author in enumerate(row_coords.index):
        if nom_est_dans_liste(canonique(author), TARGET_AUTHORS)and author != " " and not(nom_est_dans_liste(canonique(author), L)):
            x, y = row_coords.iloc[i]
            ax.text(x+0.6, y, author, color='blue', fontsize=9)
            L.append(canonique(author))
    # Tracer les topics (avec noms)
    for i  in range(NUM_TOPICS):
        x, y = col_coords.iloc[i]
        topic_names = etiquettage(NUM_TOPICS)  # Appeler la fonction etiquettage pour obtenir les noms des topics

# Maintenant vous pouvez accéder à topic_names en toute sécurité
       # topic_label = topic_names.get(topic_id, f"Topic {topic_id}")
        #if topic_label != " ":
        ax.text(x+0.6, y-0.2,topic_names[i], color='red', fontsize=10, fontweight='bold')
    # 5. Ajustement des limites automatiquement
    ax.set_xlim(0, 2.75)
    ax.set_ylim(-0.5 , 0.4)

    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    ax.set_title("Analyse en Correspondance : Auteurs vs Topics")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    """



    used_positions = []

    # Fonction pour vérifier si une position est trop proche d'une autre
    def est_trop_proche(x, y, positions, distance_min=0.3):
        for (px, py) in positions:
            if (x - px)**2 + (y - py)**2 < distance_min**2:
                return True
        return False

    # 1. Afficher les auteurs sans chevauchement
    for i, author in enumerate(row_coords.index):
        if nom_est_dans_liste(canonique(author), TARGET_AUTHORS) and author.strip() != "" and not nom_est_dans_liste(canonique(author), L):
            x, y = row_coords.iloc[i]
            orig_x = x
            while est_trop_proche(x, y, used_positions):
                x += 0.05  # Décalage si trop proche
            ax.text(x + 0.6, y, author, color='blue', fontsize=9)
            used_positions.append((x + 0.6, y))
            L.append(canonique(author))

    # 2. Afficher les topics sans chevauchement
    topic_names = etiquettage(NUM_TOPICS)  # Une seule fois, hors de la boucle
    for i in range(NUM_TOPICS):
        x, y = col_coords.iloc[i]
        label_x, label_y = x + 0.6, y - 0.2
        while est_trop_proche(label_x, label_y, used_positions):
            label_y -= 0.03
        ax.text(label_x, label_y, topic_names[i], color='red', fontsize=10, fontweight='bold')
        used_positions.append((label_x, label_y))


    ax.set_xlim(0, 2.75)
    ax.set_ylim(-0.5 , 0.4)

    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    ax.set_title("Analyse en Correspondance : Auteurs vs Topics")
    plt.grid(True)
    plt.tight_layout()
    plt.show()