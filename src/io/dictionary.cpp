// GPU-Native-CFD，GPL-3.0-or-later。見 dictionary.hpp。

#include "io/dictionary.hpp"

#include <charconv>
#include <stdexcept>

namespace gnc
{

namespace
{

bool isDirective(const Token& tok)
{
    return tok.kind == Token::WORD
        && !tok.text.empty()
        && (tok.text[0] == '#' || tok.text[0] == '$');
}

void rejectDirective(FoamTokenizer& t, const Token& tok)
{
    if (isDirective(tok) || (tok.kind == Token::WORD && tok.text == "codeStream"))
    {
        t.error
        (
            "unsupported dictionary directive '" + tok.text
          + "': expand the dictionary with foamDictionary -expand first"
        );
    }
}

} // namespace


Dictionary Dictionary::parse
(
    FoamTokenizer& t,
    const std::string& name,
    const bool braced
)
{
    Dictionary d(name);

    for (;;)
    {
        const Token key = t.next();

        if (key.kind == Token::END)
        {
            if (braced)
            {
                t.error("unexpected end of file in dictionary " + name);
            }
            break;
        }
        if (key.isPunct('}'))
        {
            if (!braced)
            {
                t.error("unmatched '}'");
            }
            break;
        }
        if (key.kind == Token::PUNCT)
        {
            t.error("expected a keyword in " + name + ", got '" + key.text + "'");
        }
        rejectDirective(t, key);

        Entry e;
        e.key = key.text;
        e.isPattern = (key.kind == Token::STRING);

        if (t.tryPunct('{'))
        {
            e.dict = std::make_shared<Dictionary>
            (
                parse(t, name + "/" + e.key, true)
            );
        }
        else
        {
            // 收集到深度 0 的 ';'
            int depth = 0;
            for (;;)
            {
                const Token v = t.next();
                if (v.kind == Token::END)
                {
                    t.error("unexpected end of file in entry " + e.key);
                }
                rejectDirective(t, v);
                if (v.kind == Token::PUNCT)
                {
                    const char c = v.text[0];
                    if (c == ';' && depth == 0)
                    {
                        break;
                    }
                    if (c == '(' || c == '[' || c == '{')
                    {
                        ++depth;
                    }
                    else if (c == ')' || c == ']' || c == '}')
                    {
                        if (--depth < 0)
                        {
                            t.error("unbalanced '" + v.text + "' in entry " + e.key);
                        }
                    }
                }
                e.stream.push_back(v);
            }
        }
        d.entries_.push_back(std::move(e));
    }

    return d;
}


const Dictionary::Entry* Dictionary::find(const std::string& key) const
{
    for (const Entry& e : entries_)
    {
        if (!e.isPattern && e.key == key)
        {
            return &e;
        }
    }
    return nullptr;
}


const Dictionary& Dictionary::subDict(const std::string& key) const
{
    const Entry* e = find(key);
    if (!e || !e->dict)
    {
        error("missing sub-dictionary '" + key + "'");
    }
    return *e->dict;
}


std::string Dictionary::getWord(const std::string& key) const
{
    const Entry* e = find(key);
    if (!e || e->dict)
    {
        error("missing entry '" + key + "'");
    }
    if
    (
        e->stream.size() != 1
     || (e->stream[0].kind != Token::WORD && e->stream[0].kind != Token::STRING)
    )
    {
        error("entry '" + key + "' is not a single word");
    }
    return e->stream[0].text;
}


std::string Dictionary::getWordOrDefault
(
    const std::string& key,
    const std::string& deflt
) const
{
    return found(key) ? getWord(key) : deflt;
}


std::int64_t Dictionary::getLabel(const std::string& key) const
{
    const std::string s = getWord(key);
    std::int64_t v = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc() || r.ptr != s.data() + s.size())
    {
        error("entry '" + key + "' is not an integer: " + s);
    }
    return v;
}


double Dictionary::getScalar(const std::string& key) const
{
    const std::string s = getWord(key);
    double v = 0;
    if (!parseScalar(s, v))
    {
        error("entry '" + key + "' is not a number: " + s);
    }
    return v;
}


void Dictionary::error(const std::string& msg) const
{
    throw std::runtime_error("dictionary " + name_ + ": " + msg);
}

} // namespace gnc
