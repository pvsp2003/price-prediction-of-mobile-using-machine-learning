#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


color_to_number = {
    "Blue": 0, "Black": 1, "Green": 2, "Yellow": 3, "White": 4, "Gray": 5, "Red": 6, "Purple": 7,
    "Greener": 8, "Silver": 9, "Bronze": 10, "Rainforest": 11, "Sunshower": 12, "Night": 13, "Gold": 14,
    "Sea": 15, "Matter": 16, "Hyperspace": 17, "Glow": 18, "Grey": 19, "BLUE": 20, "Orange": 21,
    "Midnight": 22, "Starlight": 23, "Pink": 24, "Graphite": 25, "Coral": 26, "Copper": 27, "Blush": 28,
    "Brown": 29, "Mint": 30, "Navy": 31, "Onyx": 32, "Violet": 33, "Lime": 34, "Lavender": 35,
    "Peach": 36, "Cream": 37, "blue": 38, "Aqua": 39, "Flame": 40, "Carbon": 41, "green": 42,
    "white": 43, "Aurora": 44, "BLACK": 45, "Mist": 46, "Chalk": 47, "Charcoal": 48, "Snow": 49,
    "Obsidian": 50, "Lemongrass": 51, "Hazel": 52, "Wave": 53, "Blaze": 54, "Dawn": 55, "Galaxy": 56,
    "Sky": 57, "Dream": 58, "Engine": 59, "Shimmer": 60, "waves": 61, "Cyan": 62, "Flare": 63,
    "Dazzle": 64, "Fantasy": 65, "Shadow": 66, "dream": 67, "Spark": 68, "GLOW": 69, "Jazz": 70,
    "Pearl": 71, "Symphony": 72, "Sapphire": 73, "Mauve": 74, "Magic": 75, "black": 76, "SeaBlue": 77,
    "WHITE": 78, "Dusk": 79, "mint": 80, "Blue&Copper": 81, "GREY": 82, "Steel": 83, "CYAN": 84,
    "Lily": 85, "Rose": 86, "grey": 87, "Magenta": 88, "Moonless": 89, "Champagne": 90, "Clay": 91,
    "Teal": 92, "Tide": 93, "Fog": 94, "Marble": 95, "PAC-MAN": 96, "Sierra": 97, "Forest": 98,
    "Ink": 99, "Ray": 100, "Void": 101, "Almond": 102, "Odyssey": 103, "Haze": 104, "Wood": 105,
    "Mirror": 106, "GB)": 107, "Slate": 108, "gold": 109, "Marigold": 110, "Snowfall": 111, "GRAY": 112,
    "Ocean": 113, "Diamond": 114
}
brand_to_number = {
    "POCO": 0, "realme": 1, "Realme": 2, "APPLE": 3, "Apple": 4, "SAMSUNG": 5, "OPPO": 6,
    "Google": 7, "vivo": 8, "Nothing": 9, "REDMI": 10, "Redmi": 11, "Mi": 12, "Xiaomi": 13,
    "10A": 14, "ï¿½9A": 15, "Nokia": 16, "MOTOROLA": 17, "A10E": 18, "Motorola": 19, "a": 20,
    "Moto": 21, "OnePlus": 22, "Huawei": 23, "Nexus": 24, "Alcatel": 25, "Lenovo": 26, "Infinix": 27
}
processor_to_number = {
    "Mediatek Helio A22 Processor, Upto 2.0 GHz Processor": 0,
    "Mediatek Dimensity 700 Processor": 1,
    "Helio G36 Processor": 2,
    "Mediatek Helio G85 Processor": 3,
    "Mediatek Helio G99 Processor": 4,
    "Mediatek Helio G96 Processor": 5,
    "Qualcomm Snapdragon 695 Processor": 6,
    "Mediatek Dimensity 810 Processor": 7,
    "Qualcomm Snapdragon 778G Processor": 8,
    "MediaTek Dimensity 700 Processor": 9,
    "Qualcomm Snapdragon 695 5G Processor": 10,
    "Qualcomm Snapdragon 7+ Gen 2 (4nm) Processor": 11,
    "MediaTek Helio G35 Processor": 12,
    "Qualcomm Snapdragon 730G Processor": 13,
    "MediaTek G35 Processor": 14,
    "Qualcomm Snapdragon 845 Processor": 15,
    "Qualcomm Snapdragon 870 Processor": 16,
    "Mediatek Helio G35 Processor": 17,
    "MediaTek Helio G80 Processor": 18,
    "Qualcomm Snapdragon 720G Processor": 19,
    "Qualcomm Snapdragon 860 Processor": 20,
    "Qualcomm Snapdragon 662 Processor": 21,
    "Qualcomm Snapdragon 732G Processor": 22,
    "MediaTek Dimensity 1200 Processor": 23,
    "Helio G88 Processor": 24,
    "Unisoc T612 Processor": 25,
    "Mediatek Helio G99 Octa Core Processor": 26,
    "Unisoc Tiger T612 (12 nm) Processor": 27,
    "Unisoc T612 processor Processor": 28,
    "Mediatek Dimensity 810 5G Processor": 29,
    "1 Year Domestic Warranty for Phone and 6 Months Warranty for In-Box Accessories": 30,
    "MediaTek Helio G85 Processor": 31,
    "Unisoc T610 Processor": 32,
    "Unisoc Tiger T616 Processor": 33,
    "SC9863A Processor": 34,
    "MediaTek Helio G95 Processor": 35,
    "Qualcomm Snapdragon 888 Processor": 36,
    "MediaTek Helio G96 Processor": 37,
    "MediaTek Dimensity 800U Processor": 38,
    "Octa-core Processor": 39,
    "Mediatek Dimensity 1080 5G Processor": 40,
    "1 Year Manufacturer Warranty for Phone and 6 Months Warranty for In-Box Accessories": 41,
    "1 YEAR ON MOBILE AND 6 MONTHS FOR ACCESORIES": 42,
    "12 Months Brand Warranty": 43,
    "Unisoc SC9863A/ Unisoc SC9863A1 Processor": 44,
    "Qualcomm Snapdragon 680 (SM6225) Processor": 45,
    "Qualcomm Snapdragon 855+ Processor": 46,
    "Mediatek Dimensity 920 Processor": 47,
    "12Months Brand Warranty": 48,
    "Unisoc T618 Processor": 49,
    "Dimensity 920 5G Processor Processor": 50,
    "MediaTek Dimensity 700 (MT6833) Processor": 51,
    "Snapdragon 720G Processor": 52,
    "Unisoc T610 Octa Core Processor": 53,
    "MediaTek Helio P60 Octa Core 2.0 GHz Processor": 54,
    "Qualcomm Snapdragon 680 Processor": 55,
    "MediaTek Helio G70 Processor": 56,
    "MediaTek Helio G90T Processor": 57,
    "Brand Warranty for 1 Year": 58,
    "1 Year Warranty for Phone and 6 Months Warranty for In-Box Accessories": 59,
    "Brand Warranty of 1 Year": 60,
    "Ceramic Shield": 61,
    "Water and Dust Resistant (1 meter for Upto 30 minutes, IP67)": 62,
    "iOS 13 Compatible": 63,
    "Mediatek Helio P35 Processor": 65,
    "Exynos 850 Processor": 66,
    "Exynos 1330, Octa Core Processor": 67,
    "12 Months Warranty": 68,
    "Qualcomm Snapdragon 750G Processor": 69,
    "MediaTek Helio P35 Processor": 71,
    "0 0 0 208MHz Processor": 72,
    "Qualcomm Snapdragon 8 Gen 1 Processor": 73,
    "1 Year Manufacturer Warranty": 74,
    "12 MONTHS": 75,
    "SEC S5E8535 (Exynos 1330) Processor": 76,
    "12 Months brand Warranty , Domestic Only": 77,
    "Qualcomm Snapdragon 695 (SM6375) Processor": 78,
    "Qualcomm Snapdragon 888 Octa-Core Processor": 79,
    "Dimensity 1080, Octa Core Processor": 80,
    "Exynos 1380, Octa Core Processor": 81,
    "Octa-core(EXYNOS) Processor": 82,
    "1 Year Manufacturer Warranty for Device and 6 Months Manufacturer Warranty for In-Box Accessories": 83,
    "Exynos Octa Core Processor": 84,
    "Qualcomm Snapdragon 8 Gen 2 Processor": 86,
    "1 Year of Device & 6 Months for In-Box Accessories": 87,
    "1 Year for Mobile & 6 Months for Accessories": 88,
    "Dimensity 720 5G Processor": 90,
    "1 year manufacturer warranty for device and 6 months manufacturer warranty for in-box accessories including batteries from the date of purchase": 91,
    "Exynos 9610 Processor": 92,
    "Exynos 1280 Processor": 93,
    "1 year on phone & 6 months on accessories": 94,
    "Unisoc UMS9230 Processor": 95,
    "Mediatek MT6765 Helio P35 (12nm) Processor": 96,
    "1 Year Manufacturer Warranty for Device and 6 Months Manufacturer Warranty for In-Box": 97,
    "Octa Core Processor": 98,
    "1 YEAR Brand warranty on phone, 6 months on accessories": 99,
    "Exynos 850 (S5E3830) Processor": 100,
    "12 months": 101,
    "MediaTek Helio G80 (MT6769V) Processor": 102,
    "Qualcomm Snapdragon 8+ Gen 1 Processor": 103,
    "Exynos 9810 Processor": 104,
    "MediaTek MT6739 Processor": 105,
    "Exynos 850 Octa Core Processor": 106,
    "1 Year for Mobile & 6 Months Accessories Warranty": 107,
    "Qualcomm Snapdragon 450 (SDM450-F01) Processor": 108,
    "Qualcomm SM7150 Processor": 109,
    "1 Year Manufacturer Warranty for Device and 6 Months Manufacturer Warranty for In-box Accessories": 111,
    "Qualcomm Snapdragon 450 Processor": 112,
    "12 months on phone & 6 months on accessories": 113,
    "1 year warranty on handset and 6 months warranty on the accessories.": 114,
    "Samsung Exynos 9 Octa 9611 Processor": 115,
    "MediaTek | MT6739WW quad core Processor": 116,
    "NA Processor": 117,
    "Exynos 9825 Processor": 118,
    "Qualcomm Snapdragon (SDM450-F01) Octa Core Processor": 119,
    "MediaTek Helio G35 (MT6765G) Processor": 120,
    "1 Year for Mobile and 6 Months for Accessories": 121,
    "Exynos 7870 Processor": 122,
    "Exynos 7870 Octa Core 1.6 GHz Processor": 124,
    "MediaTek Helio P35 (MT6765) Processor": 126,
    "Qualcomm SDM450-B01 Processor": 127,
    "12 Months": 128,
    "Qualcomm SM6225 Snapdragon 680 4G (6 nm) Processor": 129,
    "Mediatek Dimensity 1300 Processor": 130,
    "MTK Helio G35 Processor": 131,
    "1 Year Warranty and 6 Months Warranty for In-Box": 132,
    "Mediatek Helio P22 Processor": 133,
    "1 year warranty by brand": 134,
    "Mediatek Helio G35 (MT6765G) Processor": 135,
    "MediaTek Helio P35 Octa Core Processor": 136,
    "Mediatek Dimensity 9000+ Processor": 137,
    "MediaTek Helio P95 Processor": 138,
    "Qualcomm Snapdragon 460 Processor": 139,
    "Brand Warranty of 1 Year Available for Mobile Including Battery and 6 Months for Accessories": 140,
    "MTK P60 Octa Core 2.0 GHz Processor": 141,
    "MediaTek Helio P22 Processor": 142,
    "MediaTek Dimensity 900 Processor": 143,
    "1 YEAR": 144,
    "Brand Warranty of 1 Year Available for Mobile and Battery and 6 Months for Accessories": 145,
    "MediaTek Dimensity 1000+ (MT6889) Processor": 146,
    "Mediatek MT6750T Octa Core 1.5 GHz Processor": 147,
    "Qualcomm Snapdragon 665 Processor": 148,
    "SDM710 Processor": 149,
    "Mediatek MT6763T Octa Core 2.5 GHz Processor": 150,
    "MTK MT6771V (P70) Processor": 151,
    "Google Tensor Processor": 152,
    "Google Tensor G2 Processor": 153,
    "Tensor G2 Processor": 154,
    "Qualcomm SDM670 Snapdragon 670 (10 nm) Processor": 155,
    "Qualcomm Snapdragon 845 64-bit Processor": 156,
    "Dimensity 6020 Processor": 157,
    "Snapdragon 695 Processor": 158,
    "Mediatek Dimensity 7200 5G Processor": 159,
    "Qualcomm Snapdragon 439 Processor": 160,
    "Mediatek Dimensity 900 Processor": 161,
    "Qualcomm Snapdragon 778G 5G Mobile Platform Processor": 162,
    "Mediatek G96 Processor": 163,
    "Dimensity 9200 Processor": 164,
    "Mediatek MT6769 Helio G70 Processor": 165,
    "Mediatek Dimensity 8200 Processor": 166,
    "MTK Dimensity 700 Processor": 167,
    "Mediatek Helio G70 Processor": 168,
    "Brand Warranty of 1 Year Available for Mobile and 6 Months for In-box Accessories": 169,
    "Turbo Processor Snapdragon 695 Processor": 170,
    "MediaTek P22 Processor": 171,
    "1 year": 172,
    "MTK Helio P22 Processor": 173,
    "Mediatek Dimensity 1200 Processor": 174,
    "MediaTek P60 Octa Core 2.0 GHz Processor": 175,
    "Qualcomm Snapdragon 665 AIE Processor": 176,
    "Domestic 1 Year on Handset and 6 Months on Accessories": 177,
    "Mediatek Dimensity 9000 Processor": 178,
    "Qualcomm Snapdragon 712 AIE Octa Core 2.3GHz Processor": 179,
    "Qualcomm Snapdragon 660 AIE Processor": 180,
    "Qualcomm Snapdragon 450 Octa Core 1.8 GHz Processor": 181,
    "domestic 1 Year on Handset and 6 Months on Accessories": 182,
    "Mediatek MT6750 Octa Core 1.5GHz Processor": 183,
    "Helio P35 Processor": 184,
    "MT6762 Octa Core (Helio P22) Processor": 185,
    "Qualcomm Snapdragon 626 Processor": 186,
    "Qualcomm Snapdragon 675AIE Processor": 187,
    "1 Year for Handset and 6 Months for Inbox Accessories": 188,
    "Qualcomm Snapdragon 675AIE Octa Core 2.0GHz Processor": 189,
    "Qualcomm Snapdragon 712 AIE Processor": 190,
    "Mediatek Dimensity 800U Processor": 191,
    "1 YEAR FOR DEVICE & SIX MONTH FOR INBOX ACCESSORIES": 192,
    "Snapdragon 439 Processor": 193,
    "Domestic 1 Year of Device & 6 Months for In-Box Accessories": 194,
    "Qualcomm Snapdragon 778G+ Processor": 195,
    "1 Year Manufacturer Warranty for Handset and 6 Months Warranty for In the Box Accessories": 196,
    "Qualcomm Snapdragon 865 Processor": 197,
    "Qualcomm Snapdragon 636 Processor": 198,
    "Qualcomm Snapdragon 425 Processor": 199,
    "Snapdragon@ 8 Gen 1 Processor": 200,
    "Qualcomm Snapdragon 625 Processor": 201,
    "MediaTek Helio G25 Processor": 202,
    "Qualcomm Snapdragon 425 1.4 GHz Processor": 203,
    "Qualcomm Snapdragon 730 Processor": 204,
    "2.0 GHz Mediatek P22 Octacore Processor": 205,
    "Mediatek Helio G25 Processor": 206,
    "Qualcomm Snapdragon 632 Processor": 207,
    "Mediateck Processor": 208,
    "MSM8228 Processor": 209,
    "Qualcomm Snapdragon 625 64-bit Octa Core 2GHz Processor": 210,
    "1 Year Manufacturer Warranty": 211,
    "1 Year Manufacturer Warranty": 212,
    "Qualcomm Snapdragon 430 64-bit Octa Core 1.4GHz Processor": 213,
    "Qualcomm Snapdragon 650 64-bit Processor": 214,
    "Snapdragon 820 Kryo Processor": 215,
    "1 year on phone & 6 months on box accessories of domestic only": 216,
    "2nd-gen Snapdragon 615 64-bit octa-core Processor": 217,
    "Qualcomm Snapdragon 855 Processor": 218,
    "Qualcomm Snapdragonâ„¢ 750G Processor": 219,
    "Qualcomm Snapdragon 835 Octa Core 2.5 GHz Processor": 220,
    "Qualcomm Snapdragon 625 64 bit Octa Core 2GHz Processor": 221,
    "Qualcomm Snapdragon 625 Octa Core 2 Ghz Processor": 222,
    "Qualcomm Snapdragon 675 Processor": 223,
    "SC6531E Processor": 224,
    "MediaTek Processor": 225,
    "1 Year Manufacturer Warranty for Device and 6 Months Manufacturer Warranty for In-box Accessories Including Batteries from the Date of Purchase": 226,
    "MT6260A Processor": 227,
    "Brand Warranty of 1 Year Available for Mobile": 228,
    "1 year manufacturer warranty for device and 6 months manufacturer warranty for in-box accessories including battery": 229,
    "Brand Warranty of 1 Year Available for Mobile and 6 Months for Accessories": 230,
    "Unisoc SC9863A Processor": 231,
    "1 year manufacturer warranty for device and 6 months manufacturer warranty for in-box accessories including battery from the date of purchase": 232,
    "1 Year Manufacturer Warranty for Device and 6 Months Manufacturer Warranty for In-box Accessories Including Battery from the Date of Purchase": 233,
    "Unisoc 6531F Processor": 234,
    "Unisoc 6531E Processor": 235,
    "With a 1-year domestic replacement guarantee.": 236,
    "1 Year Manufacturer Replacement Warranty": 237,
    "MTK6261D Processor": 238,
    "1 Year Replacement Warranty for Handset and 6 Months Guarantee for Accessories": 239,
    "Unisoc T606 Processor": 240,
    "1 year replacement guarantee": 241,
    "Qualcomm Snapdragon 636 Octacore Processor": 242,
    "Unisoc Processor": 243,
    "Octa Core Processor Processor": 244,
    "Unisoc UMS9117-N Processor": 245,
    "1 year domestic replacement guarantee.": 246,
    "Spreadtrum SC9863A Processor": 247,
    "N/A Processor": 248,
    "0 0 0 0 Processor Processor": 249,
    "1 Year Manufacturer Warranty for Handset and 6 Months for Accessories": 250,
    "MediaTek Helio A22 Processor": 251,
    "Qualcomm Snapdragon S4 Processor": 252,
    "Single Core Processor": 253,
    "1 Year Manufacturer Replacement warranty": 254,
    "1 year Replacement Guarante": 255,
    "1 year domestic replacement guarantee": 256,
    "one year brand waranty": 257,
    "MediaTek Helio P60 Processor": 258,
    "1 Year Manufacturer Warranty For Device and Battery and 6 Months Manufacturer Warranty for In-box Accessories From the Date of Purchase": 259,
    "Mediatek Helio G37 Processor": 260,
    "Helio G85 Processor": 261,
    "Mediatek Dimensity 930 Processor": 262,
    "Dimensity 8020 Processor": 263,
    "UNISOC T700 Processor": 264,
    "Qualcomm Snapdragon 888 + Processor": 265,
    "Qualcomm Snapdragon 778G Plus Processor": 266,
    "Qualcomm Snapdragon 480 Pro Processor": 267,
    "Mediatek MTK6737 Quad Core 1.3Ghz Processor": 268,
    "Qualcomm Snapdragon 870 5G (SM8250-AC) Processor": 269,
    "Snapdragon 460 Processor": 270,
    "12 Month": 271,
    "Domestic warranty of 12 months on phone & 6 months on accessries": 272,
    "Domestic warranty of 12 months on phone & 6 months on accessories": 273,
    "QualcommÂ® Snapdragonâ„¢ 765G Processor": 274,
    "domestic warranty of 12 months on phone & 6 months on accessories": 275,
    "1 year Brand Warranty": 276,
    "Domestic Warranty of 12 Months on Phone & 6 Months on Accessories": 277,
    "1 year Warranty": 278,
    "1 Yera": 279,
    "12 months Warranty": 280,
    "Qualcomm Snapdragon Octa Core 750G 5G Processor Processor": 281,
    "Domestic Warranty of 12 months on phone & 6 months on accessories": 282,
    "1 Year Warranty For Phone and 6 Months Warranty for in-box Accessories": 283,
    "QualcommÂ® Snapdragonâ„¢ 855 Plus Processor": 284,
    "1 Year for  handset ,6 months fro accessories": 285,
    "QualcommÂ® Snapdragonâ„¢ 855 Plus (Octa-core, 7nm, up to 2.96 GHz) , with Qualcomm AI Engine Processor": 286,
    "12 month": 287,
    "HiSilicon Kirin 710F AI Chipset with Dual-NPU Processor": 288,
    "HUAWEI Kirin 955 ARM Cortex-A72 64-bit + ARM Cortex-A53 64-bit Processor": 289,
    "HUAWEI Kirin 955 ARM Cortex-A72 64-bit + ARM Cortex-A53 64-bit Octa Core 2.5GHz Processor": 290,
    "MediaTek MT6572 Processor": 291,
    "Kirin 710 Processor": 292,
    "Qualcomm Snapdragon 810 v2.1 64-bit Processor": 293,
    "MT6589 Processor": 294,
    "Cortex-A7 Processor": 295,
    "Qualcomm Scorpion Processor": 296,
    "MTK6737 Processor": 297,
    "MSM8952 Processor": 298,
    "18 Months": 299,
    "MSM8909 Processor": 300,
    "18 MONTHS": 301,
    "Brand Warranty of 18 Months Available for Mobile": 302,
    "MT6753 Processor": 303,
    "MTK 6592M Processor": 304,
    "MediaTek MTK Helio P25 Octa Core 2.5 GHz Processor": 305,
    "MTK8735 Processor": 306,
    "Qualcomm SDM710 Processor": 307,
    "MediaTek MT6735P 64-bit Processor": 308,
    "MediaTek 6582 Processor": 309,
    "Qualcomm Snapdragon SDM450 Processor": 310,
    "Mediatek Helio P10 64-bit Octa Core 1.8GHz Processor": 311,
    "Qualcomm Snapdragon 820 Quad Core 2.15GHz Processor": 312,
    "Qualcomm SDM855 Processor": 313,
    "MediaTek P22 Octa Core Processor": 314,
    "Qualcomm Snapdragon 430 Octa Core 1.4GHz Processor": 315,
    "Unisoc Spreadtrum SC9863A1 Processor": 316,
    "G37 Processor": 317,
    "Mediatek Helio A22 Processor": 318,
    "Dimensity 810 Processor": 319,
    "Spreadtrum SC9863A1 Processor": 320,
    "MediaTek G37 Processor": 321,
    "Meditek Helio G37 Processor": 322,
    "MediaTek Helio G88 Processor": 323,
    "Mediatek G99 Processor": 324,
    "Mediatek Dimensity 1080 Processor": 325,
    "Unisoc T616 Processor": 326,
    "MediaTek Helio A20 Processor": 327,
    "UniSoc T610 Processor": 328,
    "Mediatek Helio G88 Processor": 329,
}


# In[3]:


data = pd.read_csv('C:/Users/Asus/Desktop/outputwithoutprice3.csv')


# In[4]:


X = data.drop('classprice', axis=1)
y = data['classprice']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


model = GaussianNB()
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[10]:


print(classification_report(y_test, y_pred))


# In[11]:


user_input = input("Enter a color name: ")

# Find the corresponding number
if user_input in color_to_number:
    color_number = color_to_number[user_input]
else:
    color_number=0
user_input = input("Enter a brand name: ")

if user_input in brand_to_number:
    brand_number = brand_to_number[user_input]
else:
    brand_number=0
user_input = input("Enter a processor name: ")

if user_input in processor_to_number:
    processor_number = processor_to_number[user_input]
else:
    processor_number=0
ram = int(input("Enter RAM (in GB): "))
rom = int(input("Enter ROM/Storage (in GB): "))
back_camera = int(input("Enter Back Camera resolution (in MP): "))
front_camera = int(input("Enter Front Camera resolution (in MP): "))
battery = int(input("Enter Battery capacity (in mAh): "))
price=int(input("enter the price:"))


# In[12]:


specific_input = np.array([[color_number,brand_number,processor_number, ram, rom, back_camera,front_camera,battery]])
predicted_class = model.predict(specific_input)
print("Predicted class:", predicted_class)


# In[13]:


price_bounds = {
    "Very Low": (0, 7000),
    "Low": (7001, 15000),
    "Medium": (15001, 25000),
    "High": (25001, 50000),
    "Very High": (50001, float('inf')),  # 'inf' represents infinity
}
pricerange=price_bounds[predicted_class[0]]
if price<pricerange[0]:
    print("Very good option")
elif price>=pricerange[0] and price<pricerange[1]:
    print("Decent price")
else:
    print("Price too high! Avoid it")


# In[ ]:




