from owlready2 import *
from contextlib import suppress

onto_path.append("/home/hoyland/code/oe-ontology/")

onto = get_ontology("OEOntology.owl").load()

classes = list(onto.classes())
print(classes)