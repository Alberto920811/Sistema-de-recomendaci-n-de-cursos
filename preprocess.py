class dirty_word:
    Vec_column = []

    def __init__(self,vec_column):
        self.Vec_column = vec_column
        

    def clean_vector(self):
        vec_clean = []
        for x in self.Vec_column:
            if x != None:
                vec_clean.append(x.replace("</p><p><br></p><p>", " ")
                                  .replace("<p>"," ")
                                  .replace("</p>","")
                                  .replace("\n","")
                                  .replace("<br />","")
                                  .replace(","," ")
                                  .replace("."," ")
                                  .replace(":","")
                                  .replace("(","")
                                  .replace(")","")
                                  .replace("&shy;","")
                                  .replace("&","")
                                  .replace("acute;","")
                                  .replace("<p","")
                                  .replace('Ã³','ó')
                                  .replace('Ã\xad','í')
                                  .replace('Ã©','é')
                                  .replace('Ã¡','á')
                                  .replace('Ã±','ñ')
                                  .replace('Ãº','ú')
                                  .replace('Ã\x93','Ó')
                                  .replace('Ã\x8d','Í')
                                  .replace('Ã\x91','Ñ')
                                  .replace('Ã\x89','É')
                                  .replace('Ã\x81','Á')
                                  .replace('\t','')
                                  .replace('/',' ')
                                  .replace('-','')
                                  .replace('ï\x82·',' ')
                                  .replace('Ã©','')
                                  .replace('â\x80\x9d ','')
                                  .replace('â\x80\x93 ','')
                                  .replace('â\x80\x9c ','')
                                  .replace('â\x80\x9c','')
                                  .lower()) 
            if x == None:
                vec_clean.append("campo vacio")  
        return vec_clean
    
class stop_words:

    def __init__(self, vec):
        self.vec = vec

    def remove(self):
        stoplist = set(
                       ' un una unas unos uno sobre todo también tras otro algún alguno '
                       ' alguna algunos algunas ser es soy eres somos sois estoy esta estamos '
                       ' estais estan como en para atras porque por qué estado estaba ante '
                       ' antes siendo ambos pero por poder puede puedo podemos podeis pueden '
                       ' fui fue fuimos fueron hacer hago hace hacemos haceis hacen cada fin '
                       ' incluso primero desde conseguir consigo consigue consigues conseguimos '
                       ' consiguen ir voy va vamos vais van vaya gueno ha tener tengo tiene '
                       ' tenemos teneis tienen el la lo las los su aqui mio tuyo ellos ellas nos '
                       ' nosotros vosotros vosotras si dentro solo solamente saber sabes sabe '
                       ' sabemos sabeis saben ultimo largo bastante haces muchos aquellos al '
                       ' aquellas sus entonces tiempo verdad verdadero verdadera cierto ciertos '
                       ' cierta ciertas intentar intento intenta intentas intentamos intentais '
                       ' intentan dos bajo arriba encima usar uso usas usa usamos usais usan le '
                       ' emplear empleo empleas emplean ampleamos empleais valor muy era eras '
                       ' eramos eran modo bien cual cuando donde mientras quien con entre sin '
                       ' trabajo trabajar trabajas trabaja trabajamos trabajais trabajan podria '
                       ' podrias podriamos podrian podriais yo aquel de y a este esta estos esto '
                       ' esta esa esas eso esos ese de del está es se te tu campo vacio que están '
                       ' a about above across after afterwards again against all almost alone along '
                       ' already also although always am among amongst amoungst amount an '
                       ' and another any anyhow anyone anything anyway '
                       ' anywhere are around as at back be became because become becomes becoming been before '
                       ' beforehand behind being below beside besides between beyond bill both bottom but by call can '
                       ' cannot cant co computer con could couldnt cry de describe detail do done down due during each '
                       ' egeight either eleven else elsewhere empty enough etc even ever every everyone everything '
                       ' everywhere except few fifteen fify fill find fire first five for former formerly forty '
                       ' found four from front full further get give go had has hasnt have he hence '
                       ' her here hereafter hereby herein hereupon hers herse him himse his how however hundred i ie '
                       ' if in inc indeed interest into is it its itse keep last latter latterly least less ltd '
                       ' made many may me meanwhile might mill mine more moreover most mostly move much '
                       ' must my myse name namely neither never nevertheless next nine no nobody none noone nor not '
                       ' nothing now nowhere of off often on once one only onto or other others otherwise our '
                       ' ours ourselves out over own part per perhaps please put rather re same see '
                       ' seem seemed seeming seems serious several she should show side since sincere six '
                       ' sixty so some somehow someone something sometime sometimes somewhere '
                       ' still such system take ten than that the their them themselves then '
                       ' thence there thereafter thereby therefore therein thereupon these they thick thin '
                       ' third this those though three through throughout thru thus to together too top toward '
                       ' towards twelve twenty two un under until up upon us very via was we well were what '
                       ' whatever when whence whenever where whereafter whereas whereby '
                       ' wherein whereupon wherever whether which while whither who whoever whole whom whose '
                       ' why will with within without would yet you your yours yourself yourselves '
                       .split()) 
        
        filtered_sentence = [w for w in self.vec if w not in stoplist] 
        return filtered_sentence


    