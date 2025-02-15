#!/usr/bin/env python3

from indic_unicode_mapper import IndicUnicodeMapper

# create the mapper object
m = IndicUnicodeMapper()

# create the encoded and decoded texts
t = "கட்டடக்கலை என்பது கட்டடங்கள் மற்றும் அதன் உடல் கட்டமைப்புகளை வடிவமைத்தல், செயல்முறைத் திட்டமிடல், மற்றும் கட்டடங்கள் கட்டுவதை உள்ளடக்கியதாகும். க்ஷ்"
e = m.encode(text=t, lang='ta')
print(e)
d = m.decode(e)

# test if the original and the decoded texts are identical.
assert(t == d)

# test if there are unprocessed vowels.
assert(m.is_consistent(e, lang='ta'))

# test the rule generator
m.generate_norm_rule_tsv('/tmp/rule')

m.is_consistent('ச அள ய நகரக, ம தநகன , அ ம, ஓ, ச,  அௗ, க, எ ஒ;  உள ; ஜ அக;  ;  நக ஃஏ- ற தகரக.')

m.encode(text='பெற்றோா்கள் முறைாக குாியது பொிய நோிடையாக')

