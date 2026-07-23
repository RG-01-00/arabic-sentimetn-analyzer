"""
Arabic BERT Sentiment Analysis Engine

Features:
- Arabic / English language detection
- Arabic BERT sentiment prediction
- Positive / negative lexicon analysis
- Better negation detection
- Mixed sentiment detection
- TF-IDF keyword extraction
- Confidence and intensity calculation
- Sentence tokenization
- Basic sarcasm detection
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer


class ArabicBERTSentimentAnalyzer:

    def __init__(
        self,
        model_name="./models/best_bert_model",
        tfidf_corpus=None
    ):

        print(f"Loading model from: {model_name}")

        self.model_name = model_name

        # ============================================================
        # LOAD TOKENIZER AND MODEL
        # ============================================================

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.eval()

        # Automatically use GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        # ============================================================
        # LABEL MAPPING
        # ============================================================

        self.id2label = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

        # If the model contains its own label mapping,
        # use it when possible
        if hasattr(self.model.config, "id2label"):

            model_labels = self.model.config.id2label

            if model_labels:

                normalized_labels = {}

                for key, value in model_labels.items():

                    try:
                        key = int(key)
                    except Exception:
                        continue

                    value = str(value).lower()

                    if "neg" in value:
                        normalized_labels[key] = "negative"

                    elif "neu" in value:
                        normalized_labels[key] = "neutral"

                    elif "pos" in value:
                        normalized_labels[key] = "positive"

                if len(normalized_labels) == 3:

                    self.id2label = normalized_labels

        # ============================================================
        # ARABIC REGEX
        # ============================================================

        self.arabic_pattern = re.compile(
            r"[\u0600-\u06FF"
            r"\u0750-\u077F"
            r"\u08A0-\u08FF"
            r"\uFB50-\uFDFF"
            r"\uFE70-\uFEFF]"
        )

        self.english_pattern = re.compile(
            r"[a-zA-Z]"
        )

        # ============================================================
        # NEGATIVE WORDS
        # ============================================================

        self.negative_words = {

            "سيء",
            "سئ",
            "سيئ",
            "سَيِّئ",
            "مخيب",
            "فاشل",
            "رديء",
            "ضعيف",
            "مزعج",

            "كارثة",
            "فظيع",
            "مريع",
            "مأساة",
            "محبط",
            "مقرف",
            "بغيض",

            "سيئة",
            "سَيِّئَة",
            "فاسد",
            "متدني",
            "منخفض",
            "ضعيفة",

            "شكوى",
            "تذمر",
            "غضب",
            "غاضب",
            "استياء",
            "استيائي",

            "خيبة",
            "خائب",
            "مخذول",
            "مخز",
            "عار",

            "بطيء",
            "بطيئة",
            "متأخر",
            "إهمال",
            "مهمل",

            "سوء",
            "أسوأ",

            # Common Arabic sentiment words
            "كره",
            "أكره",
            "يكره",
            "كراهية",
            "حزين",
            "حزن",
            "مشكلة",
            "مشاكل",
            "سيئ",
            "تعبان",
            "تعب",
            "صعب",
            "صعوبة",
            "فوضى",
            "خطر",
            "خطير",
            "خسارة",
            "خسر",
            "فشل",
            "فاشل",
            "مؤلم",
            "ألم"
        }

        # ============================================================
        # POSITIVE WORDS
        # ============================================================

        self.positive_words = {

            "رائع",
            "ممتاز",
            "جيد",
            "جميل",
            "مذهل",
            "مفيد",
            "أعجبني",

            "رائعة",
            "ممتازة",
            "جميلة",
            "جيدة",

            "خرافي",
            "أسطوري",
            "مثالي",

            "عالية",
            "متميز",
            "متقدم",
            "فاخر",
            "أنيق",

            "سريع",
            "سريعة",
            "مفيدة",
            "متعاون",
            "ودود",
            "لطيف",

            "سعيد",
            "مسرور",
            "مبتهج",
            "راض",
            "راضية",
            "مرتاح",

            "أنصح",
            "أوصي",
            "أفضل",
            "أحسن",

            "استثنائي",
            "مبهر",

            # Common Arabic sentiment words
            "حب",
            "أحب",
            "يحب",
            "سعادة",
            "فرح",
            "ممتعة",
            "ممتع",
            "نجاح",
            "ناجح",
            "قوي",
            "قوية",
            "مميز",
            "مميزة",
            "مفيد",
            "مفيدة",
            "ممتعة",
            "راضي",
            "رضا",
            "تحسن",
            "تحسين"
        }

        # ============================================================
        # MULTI-WORD EXPRESSIONS
        # ============================================================

        self.negative_phrases = [

            "سيء جدا",
            "مخيب للآمال",
            "لا أنصح",
            "لا أوصي",
            "ليس جيدا",
            "ليست جيدة",
            "غير جيد",
            "غير جيدة",
            "غير مفيد",
            "غير مفيدة",
            "أسوأ تجربة",
            "تجربة سيئة"
        ]

        self.positive_phrases = [

            "رائع جدا",
            "ممتاز جدا",
            "جيد جدا",
            "أنصح به",
            "أنصح بها",
            "أوصي به",
            "أوصي بها",
            "تجربة رائعة",
            "تجربة ممتازة",
            "لا مثيل له"
        ]

        # ============================================================
        # NEGATION WORDS
        # ============================================================

        self.negation_words = {

            "لا",
            "ليس",
            "ليست",
            "لن",
            "لم",
            "ما",
            "ليسَ",
            "لست",
            "لسنا",
            "لستم",
            "لستن",
            "غير",
            "بدون",
            "من دون"
        }

        # ============================================================
        # TF-IDF
        # ============================================================

        self.tfidf_vectorizer = TfidfVectorizer(

            max_features=100,

            token_pattern=r"(?u)\b\w+\b",

            lowercase=False

        )

        self.tfidf_fitted = False

        # Fit TF-IDF using real corpus if available
        if tfidf_corpus:

            self.tfidf_vectorizer.fit(tfidf_corpus)

            self.tfidf_fitted = True

    # ================================================================
    # LANGUAGE DETECTION
    # ================================================================

    def _is_arabic(self, text):

        return bool(
            self.arabic_pattern.search(text)
        )

    def _has_english_words(self, text):

        text_without_arabic = self.arabic_pattern.sub(
            "",
            text
        )

        return bool(
            self.english_pattern.search(
                text_without_arabic
            )
        )

    # ================================================================
    # TOKENIZATION
    # ================================================================

    def _tokenize_arabic(self, text):

        """
        Tokenizes Arabic text while removing punctuation.
        """

        tokens = re.findall(
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+",
            text
        )

        return tokens

    # ================================================================
    # NORMALIZATION
    # ================================================================

    def _normalize_arabic(self, text):

        """
        Basic Arabic normalization.
        """

        # Remove tashkeel
        text = re.sub(
            r"[\u064B-\u065F\u0670]",
            "",
            text
        )

        # Normalize Alef
        text = re.sub(
            r"[إأآٱ]",
            "ا",
            text
        )

        # Normalize Yeh
        text = text.replace(
            "ى",
            "ي"
        )

        # Normalize Teh Marbuta
        text = text.replace(
            "ة",
            "ه"
        )

        # Remove tatweel
        text = text.replace(
            "ـ",
            ""
        )

        return text

    # ================================================================
    # COUNT SENTIMENT WORDS
    # ================================================================

    def _check_mixed_sentiment(self, text):

        normalized_text = self._normalize_arabic(
            text
        )

        tokens = self._tokenize_arabic(
            normalized_text
        )

        normalized_positive_words = {

            self._normalize_arabic(word)

            for word in self.positive_words

        }

        normalized_negative_words = {

            self._normalize_arabic(word)

            for word in self.negative_words

        }

        normalized_negation_words = {

            self._normalize_arabic(word)

            for word in self.negation_words

        }

        positive_count = 0

        negative_count = 0

        positive_found = []

        negative_found = []

        # ------------------------------------------------------------
        # WORD LEVEL SENTIMENT
        # ------------------------------------------------------------

        for token in tokens:

            if token in normalized_positive_words:

                positive_count += 1

                positive_found.append(token)

            elif token in normalized_negative_words:

                negative_count += 1

                negative_found.append(token)

        # ------------------------------------------------------------
        # PHRASE LEVEL SENTIMENT
        # ------------------------------------------------------------

        for phrase in self.positive_phrases:

            normalized_phrase = self._normalize_arabic(
                phrase
            )

            if normalized_phrase in normalized_text:

                positive_count += 1

                positive_found.append(
                    phrase
                )

        for phrase in self.negative_phrases:

            normalized_phrase = self._normalize_arabic(
                phrase
            )

            if normalized_phrase in normalized_text:

                negative_count += 1

                negative_found.append(
                    phrase
                )

        # ------------------------------------------------------------
        # NEGATION
        # ------------------------------------------------------------

        negation_positions = [

            index

            for index, token in enumerate(tokens)

            if token in normalized_negation_words

        ]

        has_negation = len(
            negation_positions
        ) > 0

        # ------------------------------------------------------------
        # NEGATION AROUND POSITIVE WORD
        # Example:
        #
        # "ليس جيدا"
        #
        # This is negative, not mixed.
        # ------------------------------------------------------------

        negated_positive = False

        for index, token in enumerate(tokens):

            if token in normalized_positive_words:

                previous_tokens = tokens[
                    max(0, index - 3):index
                ]

                if any(

                    negation in previous_tokens

                    for negation in normalized_negation_words

                ):

                    negated_positive = True

        # ------------------------------------------------------------
        # NEGATION AROUND NEGATIVE WORD
        # Example:
        #
        # "ليس سيئا"
        #
        # This may be positive.
        # ------------------------------------------------------------

        negated_negative = False

        for index, token in enumerate(tokens):

            if token in normalized_negative_words:

                previous_tokens = tokens[
                    max(0, index - 3):index
                ]

                if any(

                    negation in previous_tokens

                    for negation in normalized_negation_words

                ):

                    negated_negative = True

        # ------------------------------------------------------------
        # ADJUST SENTIMENT COUNTS
        # ------------------------------------------------------------

        if negated_positive:

            positive_count = max(
                0,
                positive_count - 1
            )

            negative_count += 1

        if negated_negative:

            negative_count = max(
                0,
                negative_count - 1
            )

            positive_count += 1

        is_mixed = (

            positive_count > 0

            and

            negative_count > 0

        )

        return {

            "is_mixed": is_mixed,

            "positive_count": positive_count,

            "negative_count": negative_count,

            "positive_words": positive_found,

            "negative_words": negative_found,

            "has_negation": has_negation,

            "negation_positions": negation_positions,

            "negated_positive": negated_positive,

            "negated_negative": negated_negative
        }

    # ================================================================
    # TF-IDF ANALYSIS
    # ================================================================

    def _get_tfidf_features(self, text):

        """
        Extract important words using TF-IDF.

        Important:
        TF-IDF must ideally be fitted on a real corpus.
        """

        try:

            # If no corpus was supplied, fit on this text.
            # This is only keyword extraction, not meaningful
            # corpus-level TF-IDF.
            if not self.tfidf_fitted:

                self.tfidf_vectorizer.fit(
                    [text]
                )

                self.tfidf_fitted = True

            tfidf_matrix = (

                self.tfidf_vectorizer.transform(
                    [text]
                )

            )

            feature_names = (

                self.tfidf_vectorizer
                .get_feature_names_out()

            )

            scores = (

                tfidf_matrix
                .toarray()[0]

            )

            top_indices = (

                scores.argsort()[-10:][::-1]

            )

            top_words = []

            for index in top_indices:

                if scores[index] > 0:

                    top_words.append({

                        "word": feature_names[index],

                        "score": round(
                            float(scores[index]),
                            4
                        )

                    })

            return {

                "top_words": top_words,

                "has_tfidf": True

            }

        except Exception as error:

            return {

                "top_words": [],

                "has_tfidf": False,

                "error": str(error)

            }

    # ================================================================
    # SENTIMENT INFORMATION
    # ================================================================

    def _get_sentiment_emoji(self, sentiment):

        return {

            "positive": "😊",

            "negative": "😞",

            "neutral": "😐",

            "mixed": "🤔",

            "positive_mixed": "😊🤔",

            "negative_mixed": "😞🤔",

            "error": "⚠️"

        }.get(

            sentiment,

            "📝"

        )

    def _get_sentiment_description(self, sentiment):

        return {

            "positive":
                "إيجابي - النص يحمل مشاعر إيجابية",

            "negative":
                "سلبي - النص يحمل مشاعر سلبية",

            "neutral":
                "محايد - النص لا يحمل مشاعر واضحة",

            "mixed":
                "مختلط - النص يحمل مشاعر متضاربة",

            "positive_mixed":
                "إيجابي مع بعض السلبية",

            "negative_mixed":
                "سلبي مع بعض الإيجابية",

            "error":
                "خطأ"

        }.get(

            sentiment,

            "تحليل المشاعر"

        )

    def _get_intensity(self, confidence):

        if confidence >= 0.80:

            return "قوي جداً"

        elif confidence >= 0.60:

            return "متوسط"

        elif confidence >= 0.40:

            return "ضعيف"

        else:

            return "غير مؤكد"

    # ================================================================
    # MAIN PREDICTION
    # ================================================================

    def predict(self, text):

        # ------------------------------------------------------------
        # VALIDATE INPUT
        # ------------------------------------------------------------

        if not isinstance(text, str):

            return {

                "sentiment": "error",

                "message": "Input must be a string."

            }

        text = text.strip()

        if not text:

            return {

                "sentiment": "error",

                "sentiment_emoji": "⚠️",

                "sentiment_arabic": "خطأ - النص فارغ",

                "confidence": 0.0,

                "confidence_percentage": "0%",

                "intensity": "غير صالح",

                "model": self.model_name,

                "message": "⚠️ الرجاء إدخال نص عربي.",

                "mixed": False,

                "english_detected": False

            }

        # ------------------------------------------------------------
        # LANGUAGE DETECTION
        # ------------------------------------------------------------

        has_arabic = self._is_arabic(
            text
        )

        has_english = self._has_english_words(
            text
        )

        # English only
        if has_english and not has_arabic:

            return {

                "sentiment": "error",

                "sentiment_emoji": "⚠️",

                "sentiment_arabic":
                    "خطأ - النص غير عربي",

                "confidence": 0.0,

                "score": 0.0,

                "confidence_percentage": "0%",

                "intensity": "غير صالح",

                "model": self.model_name,

                "message":
                    "⚠️ الرجاء إدخال نص عربي فقط. "
                    "هذا النموذج مدرب على تحليل المشاعر العربية.",

                "mixed": False,

                "english_detected": True,

                "summary":
                    "⚠️ النص غير عربي - يرجى إدخال نص عربي",

                "tfidf_analysis": {

                    "top_words": [],

                    "has_tfidf": False

                }

            }

        mixed_language = (

            has_arabic

            and

            has_english

        )

        # ------------------------------------------------------------
        # BERT PREDICTION
        # ------------------------------------------------------------

        inputs = self.tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            max_length=128

        )

        inputs = {

            key: value.to(self.device)

            for key, value in inputs.items()

        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        probabilities = torch.softmax(

            outputs.logits,

            dim=1

        )[0]

        sentiment_id = torch.argmax(

            probabilities

        ).item()

        bert_sentiment = self.id2label.get(

            sentiment_id,

            "neutral"

        )

        bert_confidence = float(

            probabilities[sentiment_id]

        )

        # ------------------------------------------------------------
        # RAW PROBABILITIES
        # ------------------------------------------------------------

        negative_probability = float(

            probabilities[0]

        )

        neutral_probability = float(

            probabilities[1]

        )

        positive_probability = float(

            probabilities[2]

        )

        # ------------------------------------------------------------
        # LEXICON ANALYSIS
        # ------------------------------------------------------------

        word_analysis = (

            self._check_mixed_sentiment(
                text
            )

        )

        # ------------------------------------------------------------
        # TF-IDF
        # ------------------------------------------------------------

        tfidf_analysis = (

            self._get_tfidf_features(
                text
            )

        )

        # ------------------------------------------------------------
        # FINAL SENTIMENT
        # ------------------------------------------------------------

        final_sentiment = bert_sentiment

        final_confidence = bert_confidence

        is_mixed = False

        positive_count = (

            word_analysis[
                "positive_count"
            ]

        )

        negative_count = (

            word_analysis[
                "negative_count"
            ]

        )

        # ============================================================
        # FINAL SENTIMENT LOGIC
        # ============================================================

        # ============================================================
        # RULE 1: REAL MIXED SENTIMENT
        # Both positive AND negative words exist
        # ============================================================

        if positive_count > 0 and negative_count > 0:

            final_sentiment = "mixed"

            final_confidence = bert_confidence * 0.6

            is_mixed = True


        # ============================================================
        # RULE 2: STRONG NEGATIVE LEXICON
        # Negative words exist, but no positive words
        # ============================================================

        elif negative_count > 0 and positive_count == 0:

            final_sentiment = "negative"

            # Trust BERT if it already agrees
            if bert_sentiment == "negative":

                final_confidence = bert_confidence

            else:

                # BERT disagrees with obvious negative words
                final_confidence = max(
                    bert_confidence * 0.7,
                    0.60
                )


        # ============================================================
        # RULE 3: STRONG POSITIVE LEXICON
        # Positive words exist, but no negative words
        # ============================================================

        elif positive_count > 0 and negative_count == 0:

            final_sentiment = "positive"

            if bert_sentiment == "positive":

                final_confidence = bert_confidence

            else:

                final_confidence = max(
                    bert_confidence * 0.7,
                    0.60
                )


        # ============================================================
        # RULE 4: NO CLEAR LEXICON SIGNAL
        # Trust BERT
        # ============================================================

        else:

            final_sentiment = bert_sentiment

            final_confidence = bert_confidence
                # ------------------------------------------------------------
                # LIMIT CONFIDENCE
                # ------------------------------------------------------------

            final_confidence = max(

                    0.0,

                    min(

                        1.0,

                        final_confidence

                    )

                )

        # ------------------------------------------------------------
        # PRESENTATION DATA
        # ------------------------------------------------------------

        emoji = self._get_sentiment_emoji(

            final_sentiment

        )

        description = self._get_sentiment_description(

            final_sentiment

        )

        intensity = self._get_intensity(

            final_confidence

        )

        # ------------------------------------------------------------
        # RESULT
        # ------------------------------------------------------------

        result = {

            "sentiment": final_sentiment,

            "sentiment_emoji": emoji,

            "sentiment_arabic": description,

            "confidence": round(

                final_confidence,

                3

            ),

            "score": round(

                final_confidence,

                3

            ),

            "confidence_percentage":

                f"{round(final_confidence * 100, 1)}%",

            "intensity": intensity,

            "model": self.model_name,

            "probabilities": {

                "positive":

                    f"{round(positive_probability * 100, 1)}%",

                "neutral":

                    f"{round(neutral_probability * 100, 1)}%",

                "negative":

                    f"{round(negative_probability * 100, 1)}%"

            },

            "raw_probabilities": {

                "positive":

                    round(

                        positive_probability,

                        3

                    ),

                "neutral":

                    round(

                        neutral_probability,

                        3

                    ),

                "negative":

                    round(

                        negative_probability,

                        3

                    )

            },

            "mixed": is_mixed,

            "word_analysis": {

                "positive_words_found":

                    positive_count,

                "negative_words_found":

                    negative_count,

                "positive_words":

                    word_analysis[
                        "positive_words"
                    ],

                "negative_words":

                    word_analysis[
                        "negative_words"
                    ],

                "has_negation":

                    word_analysis[
                        "has_negation"
                    ],

                "negation_positions":

                    word_analysis[
                        "negation_positions"
                    ],

                "is_mixed_sentiment":

                    word_analysis[
                        "is_mixed"
                    ]

            },

            "tfidf_analysis": tfidf_analysis,

            "summary":

                f"{emoji} {description} "

                f"(الثقة: "

                f"{round(final_confidence * 100, 1)}%)"

        }

        # ------------------------------------------------------------
        # MIXED LANGUAGE WARNING
        # ------------------------------------------------------------

        if mixed_language:

            result["warning"] = (

                "⚠️ يحتوي النص على كلمات إنجليزية. "

                "للحصول على أفضل النتائج، "

                "استخدم العربية فقط."

            )

            result["summary"] = (

                f"{emoji} "

                f"{description} "

                "⚠️ (مختلط عربي/إنجليزي)"

            )

        return result


# ====================================================================
# GLOBAL ANALYZER
# ====================================================================

_analyzer = None


def get_analyzer():

    """
    Load the model only once.

    This is important because loading BERT
    every time analyze_arabic_sentiment()
    is called is extremely inefficient.
    """


    global _analyzer
    if _analyzer is None:
        _analyzer = ArabicBERTSentimentAnalyzer(
        model_name=os.getenv(
            "MODEL_NAME",
            "./models/best_bert_model"
        )
)
    return _analyzer

# ====================================================================
# ADDITIONAL ANALYSIS FUNCTION
# ====================================================================

def analyze_arabic_sentiment(

    text,

    detailed=True

):

    """

    Complete Arabic sentiment analysis.

    The BERT model is loaded only once.
    """

    analyzer = get_analyzer()

    result = analyzer.predict(

        text

    )

    # ------------------------------------------------------------
    # TOKENIZATION
    # ------------------------------------------------------------

    tokens = analyzer._tokenize_arabic(

        text

    )

    # ------------------------------------------------------------
    # NEGATION POSITIONS
    # ------------------------------------------------------------

    negation_words = {

        analyzer._normalize_arabic(

            word

        )

        for word in analyzer.negation_words

    }

    normalized_tokens = [

        analyzer._normalize_arabic(

            token

        )

        for token in tokens

    ]

    negation_positions = [

        index

        for index, token in enumerate(

            normalized_tokens

        )

        if token in negation_words

    ]

    # ------------------------------------------------------------
    # BASIC SARCASM DETECTION
    # ------------------------------------------------------------

    sarcasm_detected = False

    sarcasm_score = 0.0

    sarcasm_indicators = [

        "يا سلام",

        "ما شاء الله",

        "رائع جدا",

        "ممتاز جدا",

        "هههه",

        "😂",

        "🤣",

        "أكيد",

        "طبعاً"

    ]

    # A very basic heuristic:
    # positive expression + negative context
    has_sarcasm_indicator = any(

        indicator in text

        for indicator in sarcasm_indicators

    )

    has_negative_words = (

        result.get(

            "word_analysis",

            {}

        ).get(

            "negative_words_found",

            0

        )

        >

        0

    )

    if (

        has_sarcasm_indicator

        and

        has_negative_words

    ):

        sarcasm_detected = True

        sarcasm_score = 0.7

    # ------------------------------------------------------------
    # ADDITIONAL DATA
    # ------------------------------------------------------------

    result["tokens"] = tokens

    result["negation_positions"] = (

        negation_positions

    )

    result["sarcasm_detected"] = (

        sarcasm_detected

    )

    result["sarcasm_score"] = (

        sarcasm_score

    )

    return result