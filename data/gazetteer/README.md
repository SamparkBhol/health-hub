# Odisha district gazetteer

`odisha_district_aliases.csv` uses project-owned stable identifiers. It does
**not** claim that those identifiers are LGD codes. Aliases cover common Census
2011 spellings and a small, manually reviewable set of Odia/Hindi forms.

Resolution is conservative: an ambiguous mention is returned as a candidate
for review, never silently forced to a district. Bhubaneswar is mapped to
Khordha only as a documented city-to-district alias for the 2011 boundary
vintage. Production must replace this small table with a versioned, stewarded
LGD/catchment crosswalk.
