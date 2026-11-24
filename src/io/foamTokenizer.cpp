// GPU-Native-CFD，GPL-3.0-or-later。見 foamTokenizer.hpp。

#include "io/foamTokenizer.hpp"

#include <cerrno>
#include <charconv>
#include <clocale>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <locale.h>

namespace gnc
{

namespace
{

bool isDigit(const char c)
{
    return c >= '0' && c <= '9';
}

bool isNumberChar(const char c)
{
    return isDigit(c) || c == '+' || c == '-' || c == '.' || c == 'E' || c == 'e';
}

bool isSpace(const char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f'
        || c == '\v';
}

bool isPunctChar(const char c)
{
    return std::strchr("(){}[];,", c) != nullptr && c != '\0';
}

} // namespace


FoamTokenizer::FoamTokenizer(const std::string& path)
:
    path_(path)
{
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is)
    {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize n = is.tellg();
    is.seekg(0);
    buf_.resize(std::size_t(n));
    if (n && !is.read(buf_.data(), n))
    {
        throw std::runtime_error("cannot read " + path);
    }
}


void FoamTokenizer::skipWs()
{
    const std::size_t n = buf_.size();
    while (pos_ < n)
    {
        const char c = buf_[pos_];
        if (isSpace(c))
        {
            ++pos_;
        }
        else if (c == '/' && pos_ + 1 < n && buf_[pos_ + 1] == '/')
        {
            while (pos_ < n && buf_[pos_] != '\n')
            {
                ++pos_;
            }
        }
        else if (c == '/' && pos_ + 1 < n && buf_[pos_ + 1] == '*')
        {
            const std::size_t start = pos_;
            pos_ += 2;
            while (pos_ + 1 < n && !(buf_[pos_] == '*' && buf_[pos_ + 1] == '/'))
            {
                ++pos_;
            }
            if (pos_ + 1 >= n)
            {
                pos_ = start;
                error("unterminated /* comment");
            }
            pos_ += 2;
        }
        else
        {
            break;
        }
    }
}


int FoamTokenizer::peekChar()
{
    skipWs();
    return pos_ < buf_.size() ? static_cast<unsigned char>(buf_[pos_]) : -1;
}


bool FoamTokenizer::atEnd()
{
    return peekChar() < 0;
}


Token FoamTokenizer::next()
{
    Token t;
    const int c = peekChar();
    if (c < 0)
    {
        return t;
    }
    if (isPunctChar(char(c)))
    {
        t.kind = Token::PUNCT;
        t.text.assign(1, char(c));
        ++pos_;
        return t;
    }
    if (c == '"')
    {
        t.kind = Token::STRING;
        ++pos_;
        while (pos_ < buf_.size() && buf_[pos_] != '"')
        {
            if (buf_[pos_] == '\\' && pos_ + 1 < buf_.size())
            {
                ++pos_;
            }
            t.text.push_back(buf_[pos_]);
            ++pos_;
        }
        if (pos_ >= buf_.size())
        {
            error("unterminated string");
        }
        ++pos_;
        return t;
    }
    t.kind = Token::WORD;
    const std::size_t start = pos_;
    if (c == '-' || c == '.' || isDigit(char(c)))
    {
        // 數字 token（ISstream.C:705–860）：之後只收數字與 + - . E e
        ++pos_;
        while (pos_ < buf_.size() && isNumberChar(buf_[pos_]))
        {
            ++pos_;
        }
    }
    else
    {
        // 單字（ISstream.C:902–930、word::valid）：括號可巢狀，例如
        // grad(U)、div(phi,k)；深度 0 的 ')' 結束單字
        unsigned depth = 0;
        while (pos_ < buf_.size())
        {
            const char ch = buf_[pos_];
            if
            (
                isSpace(ch) || ch == '"' || ch == '\'' || ch == '/'
             || ch == ';' || ch == '{' || ch == '}'
            )
            {
                break;
            }
            if (ch == '(')
            {
                ++depth;
            }
            else if (ch == ')')
            {
                if (!depth)
                {
                    break;
                }
                --depth;
            }
            ++pos_;
        }
    }
    t.text.assign(buf_.data() + start, pos_ - start);
    return t;
}


bool FoamTokenizer::tryPunct(const char c)
{
    if (peekChar() == static_cast<unsigned char>(c))
    {
        ++pos_;
        return true;
    }
    return false;
}


void FoamTokenizer::expectPunct(const char c)
{
    if (!tryPunct(c))
    {
        error(std::string("expected '") + c + "'");
    }
}


std::string FoamTokenizer::expectWord()
{
    const Token t = next();
    if (t.kind != Token::WORD)
    {
        error("expected a word, got '" + t.text + "'");
    }
    return t.text;
}


std::int64_t FoamTokenizer::readLabel()
{
    skipWs();
    const char* b = buf_.data() + pos_;
    const char* e = buf_.data() + buf_.size();
    std::int64_t v = 0;
    const auto r = std::from_chars(b, e, v);
    if (r.ec != std::errc() || r.ptr == b)
    {
        error("expected an integer");
    }
    pos_ += std::size_t(r.ptr - b);
    return v;
}


double FoamTokenizer::readScalar()
{
    skipWs();
    // 收集數字 token：開頭為 '-'、'.' 或數字，之後為數字與 + - . E e
    const std::size_t start = pos_;
    std::size_t end = pos_;
    if
    (
        end >= buf_.size()
     || !(buf_[end] == '-' || buf_[end] == '.' || isDigit(buf_[end]))
    )
    {
        error("expected a number");
    }
    ++end;
    while (end < buf_.size() && isNumberChar(buf_[end]))
    {
        ++end;
    }
    const std::string tok(buf_.data() + start, end - start);
    double v = 0;
    if (!parseScalar(tok, v))
    {
        error("invalid number '" + tok + "'");
    }
    pos_ = end;
    return v;
}


bool parseScalar(const std::string& tok, double& val)
{
    if (tok.empty())
    {
        return false;
    }

    // ISstream.C：開頭不是 '.'、其後都是數字時先當整數
    bool isIntegral = tok[0] != '.';
    for (std::size_t i = 1; i < tok.size() && isIntegral; ++i)
    {
        isIntegral = isDigit(tok[i]);
    }
    if (tok.size() == 1)
    {
        if (!isIntegral || tok[0] == '-')
        {
            return false;
        }
        val = double(tok[0] - '0');
        return true;
    }
    if (isIntegral)
    {
        const char* b = tok.data();
        const char* e = tok.data() + tok.size();
        std::int64_t i64 = 0;
        auto r = std::from_chars(b, e, i64);
        if (r.ec == std::errc() && r.ptr == e)
        {
            val = double(i64);
            return true;
        }
        if (tok[0] != '-')
        {
            std::uint64_t u64 = 0;
            r = std::from_chars(b, e, u64);
            if (r.ec == std::errc() && r.ptr == e)
            {
                val = double(u64);
                return true;
            }
        }
        // 超出整數範圍：與 OpenFOAM 相同，改用浮點轉換
    }

    // doubleScalar.cxx：ScalarConvert = ::strtold；scalarImpl.txx:103–123
    static const locale_t cLocale = newlocale(LC_ALL_MASK, "C", locale_t(0));
    char* endptr = nullptr;
    errno = 0;
    const long double parsed = strtold_l(tok.c_str(), &endptr, cLocale);
    const bool converted =
        errno == 0 && endptr != tok.c_str() && *endptr == '\0';

    constexpr long double vsmall = 1.0e-300;
    constexpr long double vgreat = 1.0e+300;
    val = (parsed >= -vsmall && parsed <= vsmall) ? 0.0 : double(parsed);

    return converted && !(parsed < -vgreat || parsed > vgreat);
}


void FoamTokenizer::readRaw(void* dst, const std::size_t n)
{
    if (pos_ + n > buf_.size())
    {
        error("binary data runs past end of file");
    }
    if (n)
    {
        std::memcpy(dst, buf_.data() + pos_, n);
    }
    pos_ += n;
}


std::size_t FoamTokenizer::line() const
{
    std::size_t l = 1;
    for (std::size_t i = 0; i < pos_ && i < buf_.size(); ++i)
    {
        if (buf_[i] == '\n')
        {
            ++l;
        }
    }
    return l;
}


void FoamTokenizer::error(const std::string& msg) const
{
    throw std::runtime_error(path_ + ":" + std::to_string(line()) + ": " + msg);
}

} // namespace gnc
