from easynmt import EasyNMT
model = EasyNMT(r"C:\Users\TEST\fxt\content\EasyNMT\omd")
print(model.translate("how are you?", target_lang="zh",source_lang="en"))