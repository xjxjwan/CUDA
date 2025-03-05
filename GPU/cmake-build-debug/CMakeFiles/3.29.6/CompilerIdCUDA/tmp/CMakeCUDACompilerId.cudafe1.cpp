#pragma section("__nv_managed_data__")
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
#line 1 "CMakeCUDACompilerId.cu"
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#define __nv_is_extended_device_lambda_with_preserved_return_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)&& defined(__nv_is_extended_device_lambda_with_preserved_return_type)
#endif

#line 1
#line 67 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
#pragma warning(push)
#pragma warning(disable: 4820)
#line 708 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\sal.h"
#pragma region Input Buffer SAL 1 compatibility macros
#line 1472
#pragma endregion Input Buffer SAL 1 compatibility macros
#line 2361 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\sal.h"
extern "C" {
#line 2971 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\sal.h"
}
#line 22 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\concurrencysal.h"
extern "C" {
#line 391 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\concurrencysal.h"
}
#line 15 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vadefs.h"
#pragma pack ( push, 8 )
#line 47 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vadefs.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 51
extern "C" {
#line 61 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vadefs.h"
typedef unsigned __int64 uintptr_t; 
#line 72 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vadefs.h"
typedef char *va_list; 
#line 155 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vadefs.h"
void __cdecl __va_start(va_list *, ...); 
#line 167 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vadefs.h"
}
#line 171 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vadefs.h"
extern "C++" {
#line 173
template< class _Ty> 
#line 174
struct __vcrt_va_list_is_reference { 
#line 176
enum: bool { __the_value}; 
#line 177
}; 
#line 179
template< class _Ty> 
#line 180
struct __vcrt_va_list_is_reference< _Ty &>  { 
#line 182
enum: bool { __the_value = '\001'}; 
#line 183
}; 
#line 185
template< class _Ty> 
#line 186
struct __vcrt_va_list_is_reference< _Ty &&>  { 
#line 188
enum: bool { __the_value = '\001'}; 
#line 189
}; 
#line 191
template< class _Ty> 
#line 192
struct __vcrt_assert_va_start_is_not_reference { 
#line 194
static_assert((!__vcrt_va_list_is_reference< _Ty> ::__the_value), "va_start argument must not have reference type and must not be parenthesized");
#line 196
}; 
#line 197
}
#line 207 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vadefs.h"
#pragma warning(pop)
#pragma pack ( pop )
#line 60 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 96 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 188 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
typedef unsigned __int64 size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#line 189
typedef __int64 ptrdiff_t; 
#line 190
typedef __int64 intptr_t; 
#line 198 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
typedef bool __vcrt_bool; 
#line 245 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
extern "C++" {
#line 247
template< class _CountofType, size_t _SizeOfArray> char (*__countof_helper(__unaligned _CountofType (& _Array)[_SizeOfArray]))[_SizeOfArray]; 
#line 251
}
#line 390 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
void __cdecl __security_init_cookie(); 
#line 399 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
void __cdecl __security_check_cookie(uintptr_t _StackCookie); 
#line 400
__declspec(noreturn) void __cdecl __report_gsfailure(uintptr_t _StackCookie); 
#line 404 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
extern uintptr_t __security_cookie; 
#line 412 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime.h"
}__pragma( pack ( pop )) 
#line 414
#pragma warning(pop)
#line 121 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 125
__pragma( pack ( push, 8 )) extern "C" {
#line 254 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt.h"
extern "C++" {
#line 256
template< bool _Enable, class _Ty> struct _CrtEnableIf; 
#line 259
template< class _Ty> 
#line 260
struct _CrtEnableIf< true, _Ty>  { 
#line 262
typedef _Ty _Type; 
#line 263
}; 
#line 264
}
#line 268 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt.h"
typedef bool __crt_bool; 
#line 371 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt.h"
void __cdecl _invalid_parameter_noinfo(); 
#line 372
__declspec(noreturn) void __cdecl _invalid_parameter_noinfo_noreturn(); 
#line 374
__declspec(noreturn) void __cdecl 
#line 375
_invoke_watson(const __wchar_t * _Expression, const __wchar_t * _FunctionName, const __wchar_t * _FileName, unsigned _LineNo, uintptr_t _Reserved); 
#line 604 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt.h"
typedef int errno_t; 
#line 605
typedef unsigned short wint_t; 
#line 606
typedef unsigned short wctype_t; 
#line 607
typedef long __time32_t; 
#line 608
typedef __int64 __time64_t; 
#line 615
typedef 
#line 610
struct __crt_locale_data_public { 
#line 612
const unsigned short *_locale_pctype; 
#line 613
int _locale_mb_cur_max; 
#line 614
unsigned _locale_lc_codepage; 
#line 615
} __crt_locale_data_public; 
#line 621
typedef 
#line 617
struct __crt_locale_pointers { 
#line 619
struct __crt_locale_data *locinfo; 
#line 620
struct __crt_multibyte_data *mbcinfo; 
#line 621
} __crt_locale_pointers; 
#line 623
typedef __crt_locale_pointers *_locale_t; 
#line 629
typedef 
#line 625
struct _Mbstatet { 
#line 627
unsigned long _Wchar; 
#line 628
unsigned short _Byte, _State; 
#line 629
} _Mbstatet; 
#line 631
typedef _Mbstatet mbstate_t; 
#line 645 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt.h"
typedef __time64_t time_t; 
#line 655 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt.h"
typedef size_t rsize_t; 
#line 2072 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt.h"
}__pragma( pack ( pop )) 
#line 2075
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wctype.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 35 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wctype.h"
const unsigned short *__cdecl __pctype_func(); 
#line 36
const wctype_t *__cdecl __pwctype_func(); 
#line 67 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wctype.h"
int __cdecl iswalnum(wint_t _C); 
#line 68
int __cdecl iswalpha(wint_t _C); 
#line 69
int __cdecl iswascii(wint_t _C); 
#line 70
int __cdecl iswblank(wint_t _C); 
#line 71
int __cdecl iswcntrl(wint_t _C); 
#line 74
int __cdecl iswdigit(wint_t _C); 
#line 76
int __cdecl iswgraph(wint_t _C); 
#line 77
int __cdecl iswlower(wint_t _C); 
#line 78
int __cdecl iswprint(wint_t _C); 
#line 79
int __cdecl iswpunct(wint_t _C); 
#line 80
int __cdecl iswspace(wint_t _C); 
#line 81
int __cdecl iswupper(wint_t _C); 
#line 82
int __cdecl iswxdigit(wint_t _C); 
#line 83
int __cdecl __iswcsymf(wint_t _C); 
#line 84
int __cdecl __iswcsym(wint_t _C); 
#line 86
int __cdecl _iswalnum_l(wint_t _C, _locale_t _Locale); 
#line 87
int __cdecl _iswalpha_l(wint_t _C, _locale_t _Locale); 
#line 88
int __cdecl _iswblank_l(wint_t _C, _locale_t _Locale); 
#line 89
int __cdecl _iswcntrl_l(wint_t _C, _locale_t _Locale); 
#line 90
int __cdecl _iswdigit_l(wint_t _C, _locale_t _Locale); 
#line 91
int __cdecl _iswgraph_l(wint_t _C, _locale_t _Locale); 
#line 92
int __cdecl _iswlower_l(wint_t _C, _locale_t _Locale); 
#line 93
int __cdecl _iswprint_l(wint_t _C, _locale_t _Locale); 
#line 94
int __cdecl _iswpunct_l(wint_t _C, _locale_t _Locale); 
#line 95
int __cdecl _iswspace_l(wint_t _C, _locale_t _Locale); 
#line 96
int __cdecl _iswupper_l(wint_t _C, _locale_t _Locale); 
#line 97
int __cdecl _iswxdigit_l(wint_t _C, _locale_t _Locale); 
#line 98
int __cdecl _iswcsymf_l(wint_t _C, _locale_t _Locale); 
#line 99
int __cdecl _iswcsym_l(wint_t _C, _locale_t _Locale); 
#line 102
wint_t __cdecl towupper(wint_t _C); 
#line 103
wint_t __cdecl towlower(wint_t _C); 
#line 104
int __cdecl iswctype(wint_t _C, wctype_t _Type); 
#line 106
wint_t __cdecl _towupper_l(wint_t _C, _locale_t _Locale); 
#line 107
wint_t __cdecl _towlower_l(wint_t _C, _locale_t _Locale); 
#line 108
int __cdecl _iswctype_l(wint_t _C, wctype_t _Type, _locale_t _Locale); 
#line 112
int __cdecl isleadbyte(int _C); 
#line 113
int __cdecl _isleadbyte_l(int _C, _locale_t _Locale); 
#line 115
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using iswctype instea" "d. See online help for details.")) int __cdecl is_wctype(wint_t _C, wctype_t _Type); 
#line 203 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wctype.h"
}__pragma( pack ( pop )) 
#line 205
#pragma warning(pop)
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\ctype.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 19
__pragma( pack ( push, 8 )) extern "C" {
#line 29
int __cdecl _isctype(int _C, int _Type); 
#line 30
int __cdecl _isctype_l(int _C, int _Type, _locale_t _Locale); 
#line 31
int __cdecl isalpha(int _C); 
#line 32
int __cdecl _isalpha_l(int _C, _locale_t _Locale); 
#line 33
int __cdecl isupper(int _C); 
#line 34
int __cdecl _isupper_l(int _C, _locale_t _Locale); 
#line 35
int __cdecl islower(int _C); 
#line 36
int __cdecl _islower_l(int _C, _locale_t _Locale); 
#line 39
int __cdecl isdigit(int _C); 
#line 41
int __cdecl _isdigit_l(int _C, _locale_t _Locale); 
#line 42
int __cdecl isxdigit(int _C); 
#line 43
int __cdecl _isxdigit_l(int _C, _locale_t _Locale); 
#line 46
int __cdecl isspace(int _C); 
#line 48
int __cdecl _isspace_l(int _C, _locale_t _Locale); 
#line 49
int __cdecl ispunct(int _C); 
#line 50
int __cdecl _ispunct_l(int _C, _locale_t _Locale); 
#line 51
int __cdecl isblank(int _C); 
#line 52
int __cdecl _isblank_l(int _C, _locale_t _Locale); 
#line 53
int __cdecl isalnum(int _C); 
#line 54
int __cdecl _isalnum_l(int _C, _locale_t _Locale); 
#line 55
int __cdecl isprint(int _C); 
#line 56
int __cdecl _isprint_l(int _C, _locale_t _Locale); 
#line 57
int __cdecl isgraph(int _C); 
#line 58
int __cdecl _isgraph_l(int _C, _locale_t _Locale); 
#line 59
int __cdecl iscntrl(int _C); 
#line 60
int __cdecl _iscntrl_l(int _C, _locale_t _Locale); 
#line 63
int __cdecl toupper(int _C); 
#line 66
int __cdecl tolower(int _C); 
#line 68
int __cdecl _tolower(int _C); 
#line 69
int __cdecl _tolower_l(int _C, _locale_t _Locale); 
#line 70
int __cdecl _toupper(int _C); 
#line 71
int __cdecl _toupper_l(int _C, _locale_t _Locale); 
#line 73
int __cdecl __isascii(int _C); 
#line 74
int __cdecl __toascii(int _C); 
#line 75
int __cdecl __iscsymf(int _C); 
#line 76
int __cdecl __iscsym(int _C); 
#line 85
__inline int __cdecl __acrt_locale_get_ctype_array_value(const unsigned short *const 
#line 86
_Locale_pctype_array, const int 
#line 87
_Char_value, const int 
#line 88
_Mask) 
#line 90
{ 
#line 96
if ((_Char_value >= (-1)) && (_Char_value <= 255)) 
#line 97
{ 
#line 98
return (_Locale_pctype_array[_Char_value]) & _Mask; 
#line 99
}  
#line 101
return 0; 
#line 102
} 
#line 124 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\ctype.h"
int __cdecl ___mb_cur_max_func(); 
#line 126
int __cdecl ___mb_cur_max_l_func(_locale_t _Locale); 
#line 152 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\ctype.h"
__forceinline int __cdecl __ascii_tolower(const int _C) 
#line 153
{ 
#line 154
if ((_C >= ('A')) && (_C <= ('Z'))) 
#line 155
{ 
#line 156
return _C - (('A') - ('a')); 
#line 157
}  
#line 158
return _C; 
#line 159
} 
#line 161
__forceinline int __cdecl __ascii_toupper(const int _C) 
#line 162
{ 
#line 163
if ((_C >= ('a')) && (_C <= ('z'))) 
#line 164
{ 
#line 165
return _C - (('a') - ('A')); 
#line 166
}  
#line 167
return _C; 
#line 168
} 
#line 170
__forceinline int __cdecl __ascii_iswalpha(const int _C) 
#line 171
{ 
#line 172
return ((_C >= ('A')) && (_C <= ('Z'))) || ((_C >= ('a')) && (_C <= ('z'))); 
#line 173
} 
#line 175
__forceinline int __cdecl __ascii_iswdigit(const int _C) 
#line 176
{ 
#line 177
return (_C >= ('0')) && (_C <= ('9')); 
#line 178
} 
#line 180
__forceinline int __cdecl __ascii_towlower(const int _C) 
#line 181
{ 
#line 182
return __ascii_tolower(_C); 
#line 183
} 
#line 185
__forceinline int __cdecl __ascii_towupper(const int _C) 
#line 186
{ 
#line 187
return __ascii_toupper(_C); 
#line 188
} 
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\ctype.h"
__inline __crt_locale_data_public *__cdecl __acrt_get_locale_data_prefix(const volatile void *const _LocalePointers) 
#line 209
{ 
#line 210
const _locale_t _TypedLocalePointers = (_locale_t)_LocalePointers; 
#line 211
return (__crt_locale_data_public *)(_TypedLocalePointers->locinfo); 
#line 212
} 
#line 218
__inline int __cdecl _chvalidchk_l(const int 
#line 219
_C, const int 
#line 220
_Mask, const _locale_t 
#line 221
_Locale) 
#line 223
{ 
#line 227
if (!_Locale) 
#line 228
{ 
#line 229
return __acrt_locale_get_ctype_array_value(__pctype_func(), _C, _Mask); 
#line 230
}  
#line 232
return __acrt_locale_get_ctype_array_value(__acrt_get_locale_data_prefix(_Locale)->_locale_pctype, _C, _Mask); 
#line 234 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\ctype.h"
} 
#line 239
__inline int __cdecl _ischartype_l(const int 
#line 240
_C, const int 
#line 241
_Mask, const _locale_t 
#line 242
_Locale) 
#line 244
{ 
#line 245
if (!_Locale) 
#line 246
{ 
#line 247
return _chvalidchk_l(_C, _Mask, 0); 
#line 248
}  
#line 250
if ((_C >= (-1)) && (_C <= 255)) 
#line 251
{ 
#line 252
return ((__acrt_get_locale_data_prefix(_Locale)->_locale_pctype)[_C]) & _Mask; 
#line 253
}  
#line 255
if ((__acrt_get_locale_data_prefix(_Locale)->_locale_mb_cur_max) > 1) 
#line 256
{ 
#line 257
return _isctype_l(_C, _Mask, _Locale); 
#line 258
}  
#line 260
return 0; 
#line 261
} 
#line 307 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\ctype.h"
}__pragma( pack ( pop )) 
#line 309
#pragma warning(pop)
#line 68 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_types.h"
#if 0
#line 68
enum cudaRoundMode { 
#line 70
cudaRoundNearest, 
#line 71
cudaRoundZero, 
#line 72
cudaRoundPosInf, 
#line 73
cudaRoundMinInf
#line 74
}; 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 104
struct char1 { 
#line 106
signed char x; 
#line 107
}; 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 109
struct uchar1 { 
#line 111
unsigned char x; 
#line 112
}; 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 115
struct __declspec(align(2)) char2 { 
#line 117
signed char x, y; 
#line 118
}; 
#endif
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 120
struct __declspec(align(2)) uchar2 { 
#line 122
unsigned char x, y; 
#line 123
}; 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 125
struct char3 { 
#line 127
signed char x, y, z; 
#line 128
}; 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 130
struct uchar3 { 
#line 132
unsigned char x, y, z; 
#line 133
}; 
#endif
#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 135
struct __declspec(align(4)) char4 { 
#line 137
signed char x, y, z, w; 
#line 138
}; 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 140
struct __declspec(align(4)) uchar4 { 
#line 142
unsigned char x, y, z, w; 
#line 143
}; 
#endif
#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 145
struct short1 { 
#line 147
short x; 
#line 148
}; 
#endif
#line 150 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 150
struct ushort1 { 
#line 152
unsigned short x; 
#line 153
}; 
#endif
#line 155 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 155
struct __declspec(align(4)) short2 { 
#line 157
short x, y; 
#line 158
}; 
#endif
#line 160 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 160
struct __declspec(align(4)) ushort2 { 
#line 162
unsigned short x, y; 
#line 163
}; 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 165
struct short3 { 
#line 167
short x, y, z; 
#line 168
}; 
#endif
#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 170
struct ushort3 { 
#line 172
unsigned short x, y, z; 
#line 173
}; 
#endif
#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 175
struct __declspec(align(8)) short4 { short x; short y; short z; short w; }; 
#endif
#line 176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 176
struct __declspec(align(8)) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 178
struct int1 { 
#line 180
int x; 
#line 181
}; 
#endif
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 183
struct uint1 { 
#line 185
unsigned x; 
#line 186
}; 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 188
struct __declspec(align(8)) int2 { int x; int y; }; 
#endif
#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 189
struct __declspec(align(8)) uint2 { unsigned x; unsigned y; }; 
#endif
#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 191
struct int3 { 
#line 193
int x, y, z; 
#line 194
}; 
#endif
#line 196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 196
struct uint3 { 
#line 198
unsigned x, y, z; 
#line 199
}; 
#endif
#line 201 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 201
struct __declspec(align(16)) int4 { 
#line 203
int x, y, z, w; 
#line 204
}; 
#endif
#line 206 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 206
struct __declspec(align(16)) uint4 { 
#line 208
unsigned x, y, z, w; 
#line 209
}; 
#endif
#line 211 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 211
struct long1 { 
#line 213
long x; 
#line 214
}; 
#endif
#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 216
struct ulong1 { 
#line 218
unsigned long x; 
#line 219
}; 
#endif
#line 222 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 222
struct __declspec(align(8)) long2 { long x; long y; }; 
#endif
#line 223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 223
struct __declspec(align(8)) ulong2 { unsigned long x; unsigned long y; }; 
#endif
#line 238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 238
struct long3 { 
#line 240
long x, y, z; 
#line 241
}; 
#endif
#line 243 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 243
struct ulong3 { 
#line 245
unsigned long x, y, z; 
#line 246
}; 
#endif
#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 248
struct __declspec(align(16)) long4 { 
#line 250
long x, y, z, w; 
#line 251
}; 
#endif
#line 253 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 253
struct __declspec(align(16)) ulong4 { 
#line 255
unsigned long x, y, z, w; 
#line 256
}; 
#endif
#line 258 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 258
struct float1 { 
#line 260
float x; 
#line 261
}; 
#endif
#line 280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 280
struct __declspec(align(8)) float2 { float x; float y; }; 
#endif
#line 285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 285
struct float3 { 
#line 287
float x, y, z; 
#line 288
}; 
#endif
#line 290 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 290
struct __declspec(align(16)) float4 { 
#line 292
float x, y, z, w; 
#line 293
}; 
#endif
#line 295 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 295
struct longlong1 { 
#line 297
__int64 x; 
#line 298
}; 
#endif
#line 300 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 300
struct ulonglong1 { 
#line 302
unsigned __int64 x; 
#line 303
}; 
#endif
#line 305 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 305
struct __declspec(align(16)) longlong2 { 
#line 307
__int64 x, y; 
#line 308
}; 
#endif
#line 310 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 310
struct __declspec(align(16)) ulonglong2 { 
#line 312
unsigned __int64 x, y; 
#line 313
}; 
#endif
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 315
struct longlong3 { 
#line 317
__int64 x, y, z; 
#line 318
}; 
#endif
#line 320 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 320
struct ulonglong3 { 
#line 322
unsigned __int64 x, y, z; 
#line 323
}; 
#endif
#line 325 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 325
struct __declspec(align(16)) longlong4 { 
#line 327
__int64 x, y, z, w; 
#line 328
}; 
#endif
#line 330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 330
struct __declspec(align(16)) ulonglong4 { 
#line 332
unsigned __int64 x, y, z, w; 
#line 333
}; 
#endif
#line 335 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 335
struct double1 { 
#line 337
double x; 
#line 338
}; 
#endif
#line 340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 340
struct __declspec(align(16)) double2 { 
#line 342
double x, y; 
#line 343
}; 
#endif
#line 345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 345
struct double3 { 
#line 347
double x, y, z; 
#line 348
}; 
#endif
#line 350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 350
struct __declspec(align(16)) double4 { 
#line 352
double x, y, z, w; 
#line 353
}; 
#endif
#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef char1 
#line 367
char1; 
#endif
#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef uchar1 
#line 368
uchar1; 
#endif
#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef char2 
#line 369
char2; 
#endif
#line 370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef uchar2 
#line 370
uchar2; 
#endif
#line 371 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef char3 
#line 371
char3; 
#endif
#line 372 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef uchar3 
#line 372
uchar3; 
#endif
#line 373 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef char4 
#line 373
char4; 
#endif
#line 374 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef uchar4 
#line 374
uchar4; 
#endif
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef short1 
#line 375
short1; 
#endif
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ushort1 
#line 376
ushort1; 
#endif
#line 377 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef short2 
#line 377
short2; 
#endif
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ushort2 
#line 378
ushort2; 
#endif
#line 379 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef short3 
#line 379
short3; 
#endif
#line 380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ushort3 
#line 380
ushort3; 
#endif
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef short4 
#line 381
short4; 
#endif
#line 382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ushort4 
#line 382
ushort4; 
#endif
#line 383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef int1 
#line 383
int1; 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef uint1 
#line 384
uint1; 
#endif
#line 385 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef int2 
#line 385
int2; 
#endif
#line 386 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef uint2 
#line 386
uint2; 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef int3 
#line 387
int3; 
#endif
#line 388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef uint3 
#line 388
uint3; 
#endif
#line 389 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef int4 
#line 389
int4; 
#endif
#line 390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef uint4 
#line 390
uint4; 
#endif
#line 391 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef long1 
#line 391
long1; 
#endif
#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ulong1 
#line 392
ulong1; 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef long2 
#line 393
long2; 
#endif
#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ulong2 
#line 394
ulong2; 
#endif
#line 395 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef long3 
#line 395
long3; 
#endif
#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ulong3 
#line 396
ulong3; 
#endif
#line 397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef long4 
#line 397
long4; 
#endif
#line 398 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ulong4 
#line 398
ulong4; 
#endif
#line 399 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef float1 
#line 399
float1; 
#endif
#line 400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef float2 
#line 400
float2; 
#endif
#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef float3 
#line 401
float3; 
#endif
#line 402 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef float4 
#line 402
float4; 
#endif
#line 403 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef longlong1 
#line 403
longlong1; 
#endif
#line 404 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ulonglong1 
#line 404
ulonglong1; 
#endif
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef longlong2 
#line 405
longlong2; 
#endif
#line 406 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ulonglong2 
#line 406
ulonglong2; 
#endif
#line 407 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef longlong3 
#line 407
longlong3; 
#endif
#line 408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ulonglong3 
#line 408
ulonglong3; 
#endif
#line 409 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef longlong4 
#line 409
longlong4; 
#endif
#line 410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef ulonglong4 
#line 410
ulonglong4; 
#endif
#line 411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef double1 
#line 411
double1; 
#endif
#line 412 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef double2 
#line 412
double2; 
#endif
#line 413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef double3 
#line 413
double3; 
#endif
#line 414 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef double4 
#line 414
double4; 
#endif
#line 426 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
#line 426
struct dim3 { 
#line 428
unsigned x, y, z; 
#line 440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
}; 
#endif
#line 442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_types.h"
#if 0
typedef dim3 
#line 442
dim3; 
#endif
#line 13 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\limits.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 16
__pragma( pack ( push, 8 )) extern "C" {
#line 74 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\limits.h"
}__pragma( pack ( pop )) 
#line 76
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stddef.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
namespace std { 
#line 25
typedef decltype(nullptr) nullptr_t; 
#line 26
}
#line 28
using std::nullptr_t;
#line 35 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stddef.h"
int *__cdecl _errno(); 
#line 38
errno_t __cdecl _set_errno(int _Value); 
#line 39
errno_t __cdecl _get_errno(int * _Value); 
#line 55 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stddef.h"
extern unsigned long __cdecl __threadid(); 
#line 57
extern uintptr_t __cdecl __threadhandle(); 
#line 61
}__pragma( pack ( pop )) 
#line 63
#pragma warning(pop)
#line 205 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 205
enum cudaError { 
#line 212
cudaSuccess, 
#line 218
cudaErrorInvalidValue, 
#line 224
cudaErrorMemoryAllocation, 
#line 230
cudaErrorInitializationError, 
#line 237
cudaErrorCudartUnloading, 
#line 244
cudaErrorProfilerDisabled, 
#line 252
cudaErrorProfilerNotInitialized, 
#line 259
cudaErrorProfilerAlreadyStarted, 
#line 266
cudaErrorProfilerAlreadyStopped, 
#line 274
cudaErrorInvalidConfiguration, 
#line 280
cudaErrorInvalidPitchValue = 12, 
#line 286
cudaErrorInvalidSymbol, 
#line 294
cudaErrorInvalidHostPointer = 16, 
#line 302
cudaErrorInvalidDevicePointer, 
#line 307
cudaErrorInvalidTexture, 
#line 313
cudaErrorInvalidTextureBinding, 
#line 320
cudaErrorInvalidChannelDescriptor, 
#line 326
cudaErrorInvalidMemcpyDirection, 
#line 336
cudaErrorAddressOfConstant, 
#line 345
cudaErrorTextureFetchFailed, 
#line 354
cudaErrorTextureNotBound, 
#line 363
cudaErrorSynchronizationError, 
#line 368
cudaErrorInvalidFilterSetting, 
#line 374
cudaErrorInvalidNormSetting, 
#line 382
cudaErrorMixedDeviceExecution, 
#line 390
cudaErrorNotYetImplemented = 31, 
#line 399
cudaErrorMemoryValueTooLarge, 
#line 405
cudaErrorStubLibrary = 34, 
#line 412
cudaErrorInsufficientDriver, 
#line 419
cudaErrorCallRequiresNewerDriver, 
#line 425
cudaErrorInvalidSurface, 
#line 431
cudaErrorDuplicateVariableName = 43, 
#line 437
cudaErrorDuplicateTextureName, 
#line 443
cudaErrorDuplicateSurfaceName, 
#line 453
cudaErrorDevicesUnavailable, 
#line 466
cudaErrorIncompatibleDriverContext = 49, 
#line 472
cudaErrorMissingConfiguration = 52, 
#line 481
cudaErrorPriorLaunchFailure, 
#line 487
cudaErrorLaunchMaxDepthExceeded = 65, 
#line 495
cudaErrorLaunchFileScopedTex, 
#line 503
cudaErrorLaunchFileScopedSurf, 
#line 519
cudaErrorSyncDepthExceeded, 
#line 531
cudaErrorLaunchPendingCountExceeded, 
#line 537
cudaErrorInvalidDeviceFunction = 98, 
#line 543
cudaErrorNoDevice = 100, 
#line 550
cudaErrorInvalidDevice, 
#line 555
cudaErrorDeviceNotLicensed, 
#line 564
cudaErrorSoftwareValidityNotEstablished, 
#line 569
cudaErrorStartupFailure = 127, 
#line 574
cudaErrorInvalidKernelImage = 200, 
#line 584
cudaErrorDeviceUninitialized, 
#line 589
cudaErrorMapBufferObjectFailed = 205, 
#line 594
cudaErrorUnmapBufferObjectFailed, 
#line 600
cudaErrorArrayIsMapped, 
#line 605
cudaErrorAlreadyMapped, 
#line 613
cudaErrorNoKernelImageForDevice, 
#line 618
cudaErrorAlreadyAcquired, 
#line 623
cudaErrorNotMapped, 
#line 629
cudaErrorNotMappedAsArray, 
#line 635
cudaErrorNotMappedAsPointer, 
#line 641
cudaErrorECCUncorrectable, 
#line 647
cudaErrorUnsupportedLimit, 
#line 653
cudaErrorDeviceAlreadyInUse, 
#line 659
cudaErrorPeerAccessUnsupported, 
#line 665
cudaErrorInvalidPtx, 
#line 670
cudaErrorInvalidGraphicsContext, 
#line 676
cudaErrorNvlinkUncorrectable, 
#line 683
cudaErrorJitCompilerNotFound, 
#line 690
cudaErrorUnsupportedPtxVersion, 
#line 697
cudaErrorJitCompilationDisabled, 
#line 702
cudaErrorUnsupportedExecAffinity, 
#line 708
cudaErrorUnsupportedDevSideSync, 
#line 719
cudaErrorContained, 
#line 724
cudaErrorInvalidSource = 300, 
#line 729
cudaErrorFileNotFound, 
#line 734
cudaErrorSharedObjectSymbolNotFound, 
#line 739
cudaErrorSharedObjectInitFailed, 
#line 744
cudaErrorOperatingSystem, 
#line 751
cudaErrorInvalidResourceHandle = 400, 
#line 757
cudaErrorIllegalState, 
#line 765
cudaErrorLossyQuery, 
#line 772
cudaErrorSymbolNotFound = 500, 
#line 780
cudaErrorNotReady = 600, 
#line 788
cudaErrorIllegalAddress = 700, 
#line 797
cudaErrorLaunchOutOfResources, 
#line 808
cudaErrorLaunchTimeout, 
#line 814
cudaErrorLaunchIncompatibleTexturing, 
#line 821
cudaErrorPeerAccessAlreadyEnabled, 
#line 828
cudaErrorPeerAccessNotEnabled, 
#line 841
cudaErrorSetOnActiveProcess = 708, 
#line 848
cudaErrorContextIsDestroyed, 
#line 855
cudaErrorAssert, 
#line 862
cudaErrorTooManyPeers, 
#line 868
cudaErrorHostMemoryAlreadyRegistered, 
#line 874
cudaErrorHostMemoryNotRegistered, 
#line 883
cudaErrorHardwareStackError, 
#line 891
cudaErrorIllegalInstruction, 
#line 900
cudaErrorMisalignedAddress, 
#line 911
cudaErrorInvalidAddressSpace, 
#line 919
cudaErrorInvalidPc, 
#line 930
cudaErrorLaunchFailure, 
#line 939
cudaErrorCooperativeLaunchTooLarge, 
#line 947
cudaErrorTensorMemoryLeak, 
#line 952
cudaErrorNotPermitted = 800, 
#line 958
cudaErrorNotSupported, 
#line 967
cudaErrorSystemNotReady, 
#line 974
cudaErrorSystemDriverMismatch, 
#line 983
cudaErrorCompatNotSupportedOnDevice, 
#line 988
cudaErrorMpsConnectionFailed, 
#line 993
cudaErrorMpsRpcFailure, 
#line 999
cudaErrorMpsServerNotReady, 
#line 1004
cudaErrorMpsMaxClientsReached, 
#line 1009
cudaErrorMpsMaxConnectionsReached, 
#line 1014
cudaErrorMpsClientTerminated, 
#line 1019
cudaErrorCdpNotSupported, 
#line 1024
cudaErrorCdpVersionMismatch, 
#line 1029
cudaErrorStreamCaptureUnsupported = 900, 
#line 1035
cudaErrorStreamCaptureInvalidated, 
#line 1041
cudaErrorStreamCaptureMerge, 
#line 1046
cudaErrorStreamCaptureUnmatched, 
#line 1052
cudaErrorStreamCaptureUnjoined, 
#line 1059
cudaErrorStreamCaptureIsolation, 
#line 1065
cudaErrorStreamCaptureImplicit, 
#line 1071
cudaErrorCapturedEvent, 
#line 1078
cudaErrorStreamCaptureWrongThread, 
#line 1083
cudaErrorTimeout, 
#line 1089
cudaErrorGraphExecUpdateFailure, 
#line 1099
cudaErrorExternalDevice, 
#line 1105
cudaErrorInvalidClusterSize, 
#line 1111
cudaErrorFunctionNotLoaded, 
#line 1117
cudaErrorInvalidResourceType, 
#line 1123
cudaErrorInvalidResourceConfiguration, 
#line 1128
cudaErrorUnknown = 999, 
#line 1136
cudaErrorApiFailureBase = 10000
#line 1137
}; 
#endif
#line 1142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1142
enum cudaChannelFormatKind { 
#line 1144
cudaChannelFormatKindSigned, 
#line 1145
cudaChannelFormatKindUnsigned, 
#line 1146
cudaChannelFormatKindFloat, 
#line 1147
cudaChannelFormatKindNone, 
#line 1148
cudaChannelFormatKindNV12, 
#line 1149
cudaChannelFormatKindUnsignedNormalized8X1, 
#line 1150
cudaChannelFormatKindUnsignedNormalized8X2, 
#line 1151
cudaChannelFormatKindUnsignedNormalized8X4, 
#line 1152
cudaChannelFormatKindUnsignedNormalized16X1, 
#line 1153
cudaChannelFormatKindUnsignedNormalized16X2, 
#line 1154
cudaChannelFormatKindUnsignedNormalized16X4, 
#line 1155
cudaChannelFormatKindSignedNormalized8X1, 
#line 1156
cudaChannelFormatKindSignedNormalized8X2, 
#line 1157
cudaChannelFormatKindSignedNormalized8X4, 
#line 1158
cudaChannelFormatKindSignedNormalized16X1, 
#line 1159
cudaChannelFormatKindSignedNormalized16X2, 
#line 1160
cudaChannelFormatKindSignedNormalized16X4, 
#line 1161
cudaChannelFormatKindUnsignedBlockCompressed1, 
#line 1162
cudaChannelFormatKindUnsignedBlockCompressed1SRGB, 
#line 1163
cudaChannelFormatKindUnsignedBlockCompressed2, 
#line 1164
cudaChannelFormatKindUnsignedBlockCompressed2SRGB, 
#line 1165
cudaChannelFormatKindUnsignedBlockCompressed3, 
#line 1166
cudaChannelFormatKindUnsignedBlockCompressed3SRGB, 
#line 1167
cudaChannelFormatKindUnsignedBlockCompressed4, 
#line 1168
cudaChannelFormatKindSignedBlockCompressed4, 
#line 1169
cudaChannelFormatKindUnsignedBlockCompressed5, 
#line 1170
cudaChannelFormatKindSignedBlockCompressed5, 
#line 1171
cudaChannelFormatKindUnsignedBlockCompressed6H, 
#line 1172
cudaChannelFormatKindSignedBlockCompressed6H, 
#line 1173
cudaChannelFormatKindUnsignedBlockCompressed7, 
#line 1174
cudaChannelFormatKindUnsignedBlockCompressed7SRGB, 
#line 1175
cudaChannelFormatKindUnsignedNormalized1010102
#line 1177
}; 
#endif
#line 1182 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1182
struct cudaChannelFormatDesc { 
#line 1184
int x; 
#line 1185
int y; 
#line 1186
int z; 
#line 1187
int w; 
#line 1188
cudaChannelFormatKind f; 
#line 1189
}; 
#endif
#line 1194 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
typedef struct cudaArray *cudaArray_t; 
#line 1199
typedef const cudaArray *cudaArray_const_t; 
#line 1201
struct cudaArray; 
#line 1206
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
#line 1211
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
#line 1213
struct cudaMipmappedArray; 
#line 1223
#if 0
#line 1223
struct cudaArraySparseProperties { 
#line 1224
struct { 
#line 1225
unsigned width; 
#line 1226
unsigned height; 
#line 1227
unsigned depth; 
#line 1228
} tileExtent; 
#line 1229
unsigned miptailFirstLevel; 
#line 1230
unsigned __int64 miptailSize; 
#line 1231
unsigned flags; 
#line 1232
unsigned reserved[4]; 
#line 1233
}; 
#endif
#line 1238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1238
struct cudaArrayMemoryRequirements { 
#line 1239
size_t size; 
#line 1240
size_t alignment; 
#line 1241
unsigned reserved[4]; 
#line 1242
}; 
#endif
#line 1247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1247
enum cudaMemoryType { 
#line 1249
cudaMemoryTypeUnregistered, 
#line 1250
cudaMemoryTypeHost, 
#line 1251
cudaMemoryTypeDevice, 
#line 1252
cudaMemoryTypeManaged
#line 1253
}; 
#endif
#line 1258 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1258
enum cudaMemcpyKind { 
#line 1260
cudaMemcpyHostToHost, 
#line 1261
cudaMemcpyHostToDevice, 
#line 1262
cudaMemcpyDeviceToHost, 
#line 1263
cudaMemcpyDeviceToDevice, 
#line 1264
cudaMemcpyDefault
#line 1265
}; 
#endif
#line 1272 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1272
struct cudaPitchedPtr { 
#line 1274
void *ptr; 
#line 1275
size_t pitch; 
#line 1276
size_t xsize; 
#line 1277
size_t ysize; 
#line 1278
}; 
#endif
#line 1285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1285
struct cudaExtent { 
#line 1287
size_t width; 
#line 1288
size_t height; 
#line 1289
size_t depth; 
#line 1290
}; 
#endif
#line 1297 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1297
struct cudaPos { 
#line 1299
size_t x; 
#line 1300
size_t y; 
#line 1301
size_t z; 
#line 1302
}; 
#endif
#line 1307 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1307
struct cudaMemcpy3DParms { 
#line 1309
cudaArray_t srcArray; 
#line 1310
cudaPos srcPos; 
#line 1311
cudaPitchedPtr srcPtr; 
#line 1313
cudaArray_t dstArray; 
#line 1314
cudaPos dstPos; 
#line 1315
cudaPitchedPtr dstPtr; 
#line 1317
cudaExtent extent; 
#line 1318
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1319
}; 
#endif
#line 1324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1324
struct cudaMemcpyNodeParams { 
#line 1325
int flags; 
#line 1326
int reserved[3]; 
#line 1327
cudaMemcpy3DParms copyParams; 
#line 1328
}; 
#endif
#line 1333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1333
struct cudaMemcpy3DPeerParms { 
#line 1335
cudaArray_t srcArray; 
#line 1336
cudaPos srcPos; 
#line 1337
cudaPitchedPtr srcPtr; 
#line 1338
int srcDevice; 
#line 1340
cudaArray_t dstArray; 
#line 1341
cudaPos dstPos; 
#line 1342
cudaPitchedPtr dstPtr; 
#line 1343
int dstDevice; 
#line 1345
cudaExtent extent; 
#line 1346
}; 
#endif
#line 1351 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1351
struct cudaMemsetParams { 
#line 1352
void *dst; 
#line 1353
size_t pitch; 
#line 1354
unsigned value; 
#line 1355
unsigned elementSize; 
#line 1356
size_t width; 
#line 1357
size_t height; 
#line 1358
}; 
#endif
#line 1363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1363
struct cudaMemsetParamsV2 { 
#line 1364
void *dst; 
#line 1365
size_t pitch; 
#line 1366
unsigned value; 
#line 1367
unsigned elementSize; 
#line 1368
size_t width; 
#line 1369
size_t height; 
#line 1370
}; 
#endif
#line 1375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1375
enum cudaAccessProperty { 
#line 1376
cudaAccessPropertyNormal, 
#line 1377
cudaAccessPropertyStreaming, 
#line 1378
cudaAccessPropertyPersisting
#line 1379
}; 
#endif
#line 1392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1392
struct cudaAccessPolicyWindow { 
#line 1393
void *base_ptr; 
#line 1394
size_t num_bytes; 
#line 1395
float hitRatio; 
#line 1396
cudaAccessProperty hitProp; 
#line 1397
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1398
}; 
#endif
#line 1410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
typedef void (__stdcall *cudaHostFn_t)(void * userData); 
#line 1415
#if 0
#line 1415
struct cudaHostNodeParams { 
#line 1416
cudaHostFn_t fn; 
#line 1417
void *userData; 
#line 1418
}; 
#endif
#line 1423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1423
struct cudaHostNodeParamsV2 { 
#line 1424
cudaHostFn_t fn; 
#line 1425
void *userData; 
#line 1426
}; 
#endif
#line 1431 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1431
enum cudaStreamCaptureStatus { 
#line 1432
cudaStreamCaptureStatusNone, 
#line 1433
cudaStreamCaptureStatusActive, 
#line 1434
cudaStreamCaptureStatusInvalidated
#line 1436
}; 
#endif
#line 1442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1442
enum cudaStreamCaptureMode { 
#line 1443
cudaStreamCaptureModeGlobal, 
#line 1444
cudaStreamCaptureModeThreadLocal, 
#line 1445
cudaStreamCaptureModeRelaxed
#line 1446
}; 
#endif
#line 1448 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1448
enum cudaSynchronizationPolicy { 
#line 1449
cudaSyncPolicyAuto = 1, 
#line 1450
cudaSyncPolicySpin, 
#line 1451
cudaSyncPolicyYield, 
#line 1452
cudaSyncPolicyBlockingSync
#line 1453
}; 
#endif
#line 1458 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1458
enum cudaClusterSchedulingPolicy { 
#line 1459
cudaClusterSchedulingPolicyDefault, 
#line 1460
cudaClusterSchedulingPolicySpread, 
#line 1461
cudaClusterSchedulingPolicyLoadBalancing
#line 1462
}; 
#endif
#line 1467 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1467
enum cudaStreamUpdateCaptureDependenciesFlags { 
#line 1468
cudaStreamAddCaptureDependencies, 
#line 1469
cudaStreamSetCaptureDependencies
#line 1470
}; 
#endif
#line 1475 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1475
enum cudaUserObjectFlags { 
#line 1476
cudaUserObjectNoDestructorSync = 1
#line 1477
}; 
#endif
#line 1482 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1482
enum cudaUserObjectRetainFlags { 
#line 1483
cudaGraphUserObjectMove = 1
#line 1484
}; 
#endif
#line 1489 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
struct cudaGraphicsResource; 
#line 1494
#if 0
#line 1494
enum cudaGraphicsRegisterFlags { 
#line 1496
cudaGraphicsRegisterFlagsNone, 
#line 1497
cudaGraphicsRegisterFlagsReadOnly, 
#line 1498
cudaGraphicsRegisterFlagsWriteDiscard, 
#line 1499
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
#line 1500
cudaGraphicsRegisterFlagsTextureGather = 8
#line 1501
}; 
#endif
#line 1506 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1506
enum cudaGraphicsMapFlags { 
#line 1508
cudaGraphicsMapFlagsNone, 
#line 1509
cudaGraphicsMapFlagsReadOnly, 
#line 1510
cudaGraphicsMapFlagsWriteDiscard
#line 1511
}; 
#endif
#line 1516 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1516
enum cudaGraphicsCubeFace { 
#line 1518
cudaGraphicsCubeFacePositiveX, 
#line 1519
cudaGraphicsCubeFaceNegativeX, 
#line 1520
cudaGraphicsCubeFacePositiveY, 
#line 1521
cudaGraphicsCubeFaceNegativeY, 
#line 1522
cudaGraphicsCubeFacePositiveZ, 
#line 1523
cudaGraphicsCubeFaceNegativeZ
#line 1524
}; 
#endif
#line 1529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1529
enum cudaResourceType { 
#line 1531
cudaResourceTypeArray, 
#line 1532
cudaResourceTypeMipmappedArray, 
#line 1533
cudaResourceTypeLinear, 
#line 1534
cudaResourceTypePitch2D
#line 1535
}; 
#endif
#line 1540 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1540
enum cudaResourceViewFormat { 
#line 1542
cudaResViewFormatNone, 
#line 1543
cudaResViewFormatUnsignedChar1, 
#line 1544
cudaResViewFormatUnsignedChar2, 
#line 1545
cudaResViewFormatUnsignedChar4, 
#line 1546
cudaResViewFormatSignedChar1, 
#line 1547
cudaResViewFormatSignedChar2, 
#line 1548
cudaResViewFormatSignedChar4, 
#line 1549
cudaResViewFormatUnsignedShort1, 
#line 1550
cudaResViewFormatUnsignedShort2, 
#line 1551
cudaResViewFormatUnsignedShort4, 
#line 1552
cudaResViewFormatSignedShort1, 
#line 1553
cudaResViewFormatSignedShort2, 
#line 1554
cudaResViewFormatSignedShort4, 
#line 1555
cudaResViewFormatUnsignedInt1, 
#line 1556
cudaResViewFormatUnsignedInt2, 
#line 1557
cudaResViewFormatUnsignedInt4, 
#line 1558
cudaResViewFormatSignedInt1, 
#line 1559
cudaResViewFormatSignedInt2, 
#line 1560
cudaResViewFormatSignedInt4, 
#line 1561
cudaResViewFormatHalf1, 
#line 1562
cudaResViewFormatHalf2, 
#line 1563
cudaResViewFormatHalf4, 
#line 1564
cudaResViewFormatFloat1, 
#line 1565
cudaResViewFormatFloat2, 
#line 1566
cudaResViewFormatFloat4, 
#line 1567
cudaResViewFormatUnsignedBlockCompressed1, 
#line 1568
cudaResViewFormatUnsignedBlockCompressed2, 
#line 1569
cudaResViewFormatUnsignedBlockCompressed3, 
#line 1570
cudaResViewFormatUnsignedBlockCompressed4, 
#line 1571
cudaResViewFormatSignedBlockCompressed4, 
#line 1572
cudaResViewFormatUnsignedBlockCompressed5, 
#line 1573
cudaResViewFormatSignedBlockCompressed5, 
#line 1574
cudaResViewFormatUnsignedBlockCompressed6H, 
#line 1575
cudaResViewFormatSignedBlockCompressed6H, 
#line 1576
cudaResViewFormatUnsignedBlockCompressed7
#line 1577
}; 
#endif
#line 1582 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1582
struct cudaResourceDesc { 
#line 1583
cudaResourceType resType; 
#line 1585
union { 
#line 1586
struct { 
#line 1587
cudaArray_t array; 
#line 1588
} array; 
#line 1589
struct { 
#line 1590
cudaMipmappedArray_t mipmap; 
#line 1591
} mipmap; 
#line 1592
struct { 
#line 1593
void *devPtr; 
#line 1594
cudaChannelFormatDesc desc; 
#line 1595
size_t sizeInBytes; 
#line 1596
} linear; 
#line 1597
struct { 
#line 1598
void *devPtr; 
#line 1599
cudaChannelFormatDesc desc; 
#line 1600
size_t width; 
#line 1601
size_t height; 
#line 1602
size_t pitchInBytes; 
#line 1603
} pitch2D; 
#line 1604
} res; 
#line 1605
}; 
#endif
#line 1610 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1610
struct cudaResourceViewDesc { 
#line 1612
cudaResourceViewFormat format; 
#line 1613
size_t width; 
#line 1614
size_t height; 
#line 1615
size_t depth; 
#line 1616
unsigned firstMipmapLevel; 
#line 1617
unsigned lastMipmapLevel; 
#line 1618
unsigned firstLayer; 
#line 1619
unsigned lastLayer; 
#line 1620
}; 
#endif
#line 1625 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1625
struct cudaPointerAttributes { 
#line 1631
cudaMemoryType type; 
#line 1642
int device; 
#line 1648
void *devicePointer; 
#line 1657
void *hostPointer; 
#line 1658
}; 
#endif
#line 1663 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1663
struct cudaFuncAttributes { 
#line 1670
size_t sharedSizeBytes; 
#line 1676
size_t constSizeBytes; 
#line 1681
size_t localSizeBytes; 
#line 1688
int maxThreadsPerBlock; 
#line 1693
int numRegs; 
#line 1700
int ptxVersion; 
#line 1707
int binaryVersion; 
#line 1713
int cacheModeCA; 
#line 1720
int maxDynamicSharedSizeBytes; 
#line 1729
int preferredShmemCarveout; 
#line 1735
int clusterDimMustBeSet; 
#line 1746
int requiredClusterWidth; 
#line 1747
int requiredClusterHeight; 
#line 1748
int requiredClusterDepth; 
#line 1754
int clusterSchedulingPolicyPreference; 
#line 1776
int nonPortableClusterSizeAllowed; 
#line 1781
int reserved[16]; 
#line 1782
}; 
#endif
#line 1787 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1787
enum cudaFuncAttribute { 
#line 1789
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
#line 1790
cudaFuncAttributePreferredSharedMemoryCarveout, 
#line 1791
cudaFuncAttributeClusterDimMustBeSet, 
#line 1792
cudaFuncAttributeRequiredClusterWidth, 
#line 1793
cudaFuncAttributeRequiredClusterHeight, 
#line 1794
cudaFuncAttributeRequiredClusterDepth, 
#line 1795
cudaFuncAttributeNonPortableClusterSizeAllowed, 
#line 1796
cudaFuncAttributeClusterSchedulingPolicyPreference, 
#line 1797
cudaFuncAttributeMax
#line 1798
}; 
#endif
#line 1803 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1803
enum cudaFuncCache { 
#line 1805
cudaFuncCachePreferNone, 
#line 1806
cudaFuncCachePreferShared, 
#line 1807
cudaFuncCachePreferL1, 
#line 1808
cudaFuncCachePreferEqual
#line 1809
}; 
#endif
#line 1815 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1815
enum cudaSharedMemConfig { 
#line 1817
cudaSharedMemBankSizeDefault, 
#line 1818
cudaSharedMemBankSizeFourByte, 
#line 1819
cudaSharedMemBankSizeEightByte
#line 1820
}; 
#endif
#line 1825 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1825
enum cudaSharedCarveout { 
#line 1826
cudaSharedmemCarveoutDefault = (-1), 
#line 1827
cudaSharedmemCarveoutMaxShared = 100, 
#line 1828
cudaSharedmemCarveoutMaxL1 = 0
#line 1829
}; 
#endif
#line 1834 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1834
enum cudaComputeMode { 
#line 1836
cudaComputeModeDefault, 
#line 1837
cudaComputeModeExclusive, 
#line 1838
cudaComputeModeProhibited, 
#line 1839
cudaComputeModeExclusiveProcess
#line 1840
}; 
#endif
#line 1845 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1845
enum cudaLimit { 
#line 1847
cudaLimitStackSize, 
#line 1848
cudaLimitPrintfFifoSize, 
#line 1849
cudaLimitMallocHeapSize, 
#line 1850
cudaLimitDevRuntimeSyncDepth, 
#line 1851
cudaLimitDevRuntimePendingLaunchCount, 
#line 1852
cudaLimitMaxL2FetchGranularity, 
#line 1853
cudaLimitPersistingL2CacheSize
#line 1854
}; 
#endif
#line 1859 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1859
enum cudaMemoryAdvise { 
#line 1861
cudaMemAdviseSetReadMostly = 1, 
#line 1862
cudaMemAdviseUnsetReadMostly, 
#line 1863
cudaMemAdviseSetPreferredLocation, 
#line 1864
cudaMemAdviseUnsetPreferredLocation, 
#line 1865
cudaMemAdviseSetAccessedBy, 
#line 1866
cudaMemAdviseUnsetAccessedBy
#line 1867
}; 
#endif
#line 1872 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1872
enum cudaMemRangeAttribute { 
#line 1874
cudaMemRangeAttributeReadMostly = 1, 
#line 1875
cudaMemRangeAttributePreferredLocation, 
#line 1876
cudaMemRangeAttributeAccessedBy, 
#line 1877
cudaMemRangeAttributeLastPrefetchLocation, 
#line 1878
cudaMemRangeAttributePreferredLocationType, 
#line 1879
cudaMemRangeAttributePreferredLocationId, 
#line 1880
cudaMemRangeAttributeLastPrefetchLocationType, 
#line 1881
cudaMemRangeAttributeLastPrefetchLocationId
#line 1882
}; 
#endif
#line 1887 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1887
enum cudaFlushGPUDirectRDMAWritesOptions { 
#line 1888
cudaFlushGPUDirectRDMAWritesOptionHost = (1 << 0), 
#line 1889
cudaFlushGPUDirectRDMAWritesOptionMemOps
#line 1890
}; 
#endif
#line 1895 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1895
enum cudaGPUDirectRDMAWritesOrdering { 
#line 1896
cudaGPUDirectRDMAWritesOrderingNone, 
#line 1897
cudaGPUDirectRDMAWritesOrderingOwner = 100, 
#line 1898
cudaGPUDirectRDMAWritesOrderingAllDevices = 200
#line 1899
}; 
#endif
#line 1904 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1904
enum cudaFlushGPUDirectRDMAWritesScope { 
#line 1905
cudaFlushGPUDirectRDMAWritesToOwner = 100, 
#line 1906
cudaFlushGPUDirectRDMAWritesToAllDevices = 200
#line 1907
}; 
#endif
#line 1912 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1912
enum cudaFlushGPUDirectRDMAWritesTarget { 
#line 1913
cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
#line 1914
}; 
#endif
#line 1920 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 1920
enum cudaDeviceAttr { 
#line 1922
cudaDevAttrMaxThreadsPerBlock = 1, 
#line 1923
cudaDevAttrMaxBlockDimX, 
#line 1924
cudaDevAttrMaxBlockDimY, 
#line 1925
cudaDevAttrMaxBlockDimZ, 
#line 1926
cudaDevAttrMaxGridDimX, 
#line 1927
cudaDevAttrMaxGridDimY, 
#line 1928
cudaDevAttrMaxGridDimZ, 
#line 1929
cudaDevAttrMaxSharedMemoryPerBlock, 
#line 1930
cudaDevAttrTotalConstantMemory, 
#line 1931
cudaDevAttrWarpSize, 
#line 1932
cudaDevAttrMaxPitch, 
#line 1933
cudaDevAttrMaxRegistersPerBlock, 
#line 1934
cudaDevAttrClockRate, 
#line 1935
cudaDevAttrTextureAlignment, 
#line 1936
cudaDevAttrGpuOverlap, 
#line 1937
cudaDevAttrMultiProcessorCount, 
#line 1938
cudaDevAttrKernelExecTimeout, 
#line 1939
cudaDevAttrIntegrated, 
#line 1940
cudaDevAttrCanMapHostMemory, 
#line 1941
cudaDevAttrComputeMode, 
#line 1942
cudaDevAttrMaxTexture1DWidth, 
#line 1943
cudaDevAttrMaxTexture2DWidth, 
#line 1944
cudaDevAttrMaxTexture2DHeight, 
#line 1945
cudaDevAttrMaxTexture3DWidth, 
#line 1946
cudaDevAttrMaxTexture3DHeight, 
#line 1947
cudaDevAttrMaxTexture3DDepth, 
#line 1948
cudaDevAttrMaxTexture2DLayeredWidth, 
#line 1949
cudaDevAttrMaxTexture2DLayeredHeight, 
#line 1950
cudaDevAttrMaxTexture2DLayeredLayers, 
#line 1951
cudaDevAttrSurfaceAlignment, 
#line 1952
cudaDevAttrConcurrentKernels, 
#line 1953
cudaDevAttrEccEnabled, 
#line 1954
cudaDevAttrPciBusId, 
#line 1955
cudaDevAttrPciDeviceId, 
#line 1956
cudaDevAttrTccDriver, 
#line 1957
cudaDevAttrMemoryClockRate, 
#line 1958
cudaDevAttrGlobalMemoryBusWidth, 
#line 1959
cudaDevAttrL2CacheSize, 
#line 1960
cudaDevAttrMaxThreadsPerMultiProcessor, 
#line 1961
cudaDevAttrAsyncEngineCount, 
#line 1962
cudaDevAttrUnifiedAddressing, 
#line 1963
cudaDevAttrMaxTexture1DLayeredWidth, 
#line 1964
cudaDevAttrMaxTexture1DLayeredLayers, 
#line 1965
cudaDevAttrMaxTexture2DGatherWidth = 45, 
#line 1966
cudaDevAttrMaxTexture2DGatherHeight, 
#line 1967
cudaDevAttrMaxTexture3DWidthAlt, 
#line 1968
cudaDevAttrMaxTexture3DHeightAlt, 
#line 1969
cudaDevAttrMaxTexture3DDepthAlt, 
#line 1970
cudaDevAttrPciDomainId, 
#line 1971
cudaDevAttrTexturePitchAlignment, 
#line 1972
cudaDevAttrMaxTextureCubemapWidth, 
#line 1973
cudaDevAttrMaxTextureCubemapLayeredWidth, 
#line 1974
cudaDevAttrMaxTextureCubemapLayeredLayers, 
#line 1975
cudaDevAttrMaxSurface1DWidth, 
#line 1976
cudaDevAttrMaxSurface2DWidth, 
#line 1977
cudaDevAttrMaxSurface2DHeight, 
#line 1978
cudaDevAttrMaxSurface3DWidth, 
#line 1979
cudaDevAttrMaxSurface3DHeight, 
#line 1980
cudaDevAttrMaxSurface3DDepth, 
#line 1981
cudaDevAttrMaxSurface1DLayeredWidth, 
#line 1982
cudaDevAttrMaxSurface1DLayeredLayers, 
#line 1983
cudaDevAttrMaxSurface2DLayeredWidth, 
#line 1984
cudaDevAttrMaxSurface2DLayeredHeight, 
#line 1985
cudaDevAttrMaxSurface2DLayeredLayers, 
#line 1986
cudaDevAttrMaxSurfaceCubemapWidth, 
#line 1987
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
#line 1988
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
#line 1989
cudaDevAttrMaxTexture1DLinearWidth, 
#line 1990
cudaDevAttrMaxTexture2DLinearWidth, 
#line 1991
cudaDevAttrMaxTexture2DLinearHeight, 
#line 1992
cudaDevAttrMaxTexture2DLinearPitch, 
#line 1993
cudaDevAttrMaxTexture2DMipmappedWidth, 
#line 1994
cudaDevAttrMaxTexture2DMipmappedHeight, 
#line 1995
cudaDevAttrComputeCapabilityMajor, 
#line 1996
cudaDevAttrComputeCapabilityMinor, 
#line 1997
cudaDevAttrMaxTexture1DMipmappedWidth, 
#line 1998
cudaDevAttrStreamPrioritiesSupported, 
#line 1999
cudaDevAttrGlobalL1CacheSupported, 
#line 2000
cudaDevAttrLocalL1CacheSupported, 
#line 2001
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
#line 2002
cudaDevAttrMaxRegistersPerMultiprocessor, 
#line 2003
cudaDevAttrManagedMemory, 
#line 2004
cudaDevAttrIsMultiGpuBoard, 
#line 2005
cudaDevAttrMultiGpuBoardGroupID, 
#line 2006
cudaDevAttrHostNativeAtomicSupported, 
#line 2007
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
#line 2008
cudaDevAttrPageableMemoryAccess, 
#line 2009
cudaDevAttrConcurrentManagedAccess, 
#line 2010
cudaDevAttrComputePreemptionSupported, 
#line 2011
cudaDevAttrCanUseHostPointerForRegisteredMem, 
#line 2012
cudaDevAttrReserved92, 
#line 2013
cudaDevAttrReserved93, 
#line 2014
cudaDevAttrReserved94, 
#line 2015
cudaDevAttrCooperativeLaunch, 
#line 2016
cudaDevAttrCooperativeMultiDeviceLaunch, 
#line 2017
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
#line 2018
cudaDevAttrCanFlushRemoteWrites, 
#line 2019
cudaDevAttrHostRegisterSupported, 
#line 2020
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
#line 2021
cudaDevAttrDirectManagedMemAccessFromHost, 
#line 2022
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
#line 2023
cudaDevAttrMaxPersistingL2CacheSize = 108, 
#line 2024
cudaDevAttrMaxAccessPolicyWindowSize, 
#line 2025
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
#line 2026
cudaDevAttrSparseCudaArraySupported, 
#line 2027
cudaDevAttrHostRegisterReadOnlySupported, 
#line 2028
cudaDevAttrTimelineSemaphoreInteropSupported, 
#line 2029
cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114, 
#line 2030
cudaDevAttrMemoryPoolsSupported, 
#line 2031
cudaDevAttrGPUDirectRDMASupported, 
#line 2032
cudaDevAttrGPUDirectRDMAFlushWritesOptions, 
#line 2033
cudaDevAttrGPUDirectRDMAWritesOrdering, 
#line 2034
cudaDevAttrMemoryPoolSupportedHandleTypes, 
#line 2035
cudaDevAttrClusterLaunch, 
#line 2036
cudaDevAttrDeferredMappingCudaArraySupported, 
#line 2037
cudaDevAttrReserved122, 
#line 2038
cudaDevAttrReserved123, 
#line 2039
cudaDevAttrReserved124, 
#line 2040
cudaDevAttrIpcEventSupport, 
#line 2041
cudaDevAttrMemSyncDomainCount, 
#line 2042
cudaDevAttrReserved127, 
#line 2043
cudaDevAttrReserved128, 
#line 2044
cudaDevAttrReserved129, 
#line 2045
cudaDevAttrNumaConfig, 
#line 2046
cudaDevAttrNumaId, 
#line 2047
cudaDevAttrReserved132, 
#line 2048
cudaDevAttrMpsEnabled, 
#line 2049
cudaDevAttrHostNumaId, 
#line 2050
cudaDevAttrD3D12CigSupported, 
#line 2051
cudaDevAttrGpuPciDeviceId = 139, 
#line 2052
cudaDevAttrGpuPciSubsystemId, 
#line 2053
cudaDevAttrHostNumaMultinodeIpcSupported = 143, 
#line 2054
cudaDevAttrMax
#line 2055
}; 
#endif
#line 2060 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2060
enum cudaMemPoolAttr { 
#line 2070
cudaMemPoolReuseFollowEventDependencies = 1, 
#line 2077
cudaMemPoolReuseAllowOpportunistic, 
#line 2085
cudaMemPoolReuseAllowInternalDependencies, 
#line 2096
cudaMemPoolAttrReleaseThreshold, 
#line 2102
cudaMemPoolAttrReservedMemCurrent, 
#line 2109
cudaMemPoolAttrReservedMemHigh, 
#line 2115
cudaMemPoolAttrUsedMemCurrent, 
#line 2122
cudaMemPoolAttrUsedMemHigh
#line 2123
}; 
#endif
#line 2128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2128
enum cudaMemLocationType { 
#line 2129
cudaMemLocationTypeInvalid, 
#line 2130
cudaMemLocationTypeDevice, 
#line 2131
cudaMemLocationTypeHost, 
#line 2132
cudaMemLocationTypeHostNuma, 
#line 2133
cudaMemLocationTypeHostNumaCurrent
#line 2134
}; 
#endif
#line 2142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2142
struct cudaMemLocation { 
#line 2143
cudaMemLocationType type; 
#line 2144
int id; 
#line 2145
}; 
#endif
#line 2150 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2150
enum cudaMemAccessFlags { 
#line 2151
cudaMemAccessFlagsProtNone, 
#line 2152
cudaMemAccessFlagsProtRead, 
#line 2153
cudaMemAccessFlagsProtReadWrite = 3
#line 2154
}; 
#endif
#line 2159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2159
struct cudaMemAccessDesc { 
#line 2160
cudaMemLocation location; 
#line 2161
cudaMemAccessFlags flags; 
#line 2162
}; 
#endif
#line 2167 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2167
enum cudaMemAllocationType { 
#line 2168
cudaMemAllocationTypeInvalid, 
#line 2172
cudaMemAllocationTypePinned, 
#line 2173
cudaMemAllocationTypeMax = 2147483647
#line 2174
}; 
#endif
#line 2179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2179
enum cudaMemAllocationHandleType { 
#line 2180
cudaMemHandleTypeNone, 
#line 2181
cudaMemHandleTypePosixFileDescriptor, 
#line 2182
cudaMemHandleTypeWin32, 
#line 2183
cudaMemHandleTypeWin32Kmt = 4, 
#line 2184
cudaMemHandleTypeFabric = 8
#line 2185
}; 
#endif
#line 2196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2196
struct cudaMemPoolProps { 
#line 2197
cudaMemAllocationType allocType; 
#line 2198
cudaMemAllocationHandleType handleTypes; 
#line 2199
cudaMemLocation location; 
#line 2206
void *win32SecurityAttributes; 
#line 2207
size_t maxSize; 
#line 2208
unsigned short usage; 
#line 2209
unsigned char reserved[54]; 
#line 2210
}; 
#endif
#line 2215 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2215
struct cudaMemPoolPtrExportData { 
#line 2216
unsigned char reserved[64]; 
#line 2217
}; 
#endif
#line 2222 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2222
struct cudaMemAllocNodeParams { 
#line 2227
cudaMemPoolProps poolProps; 
#line 2228
const cudaMemAccessDesc *accessDescs; 
#line 2229
size_t accessDescCount; 
#line 2230
size_t bytesize; 
#line 2231
void *dptr; 
#line 2232
}; 
#endif
#line 2237 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2237
struct cudaMemAllocNodeParamsV2 { 
#line 2242
cudaMemPoolProps poolProps; 
#line 2243
const cudaMemAccessDesc *accessDescs; 
#line 2244
size_t accessDescCount; 
#line 2245
size_t bytesize; 
#line 2246
void *dptr; 
#line 2247
}; 
#endif
#line 2252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2252
struct cudaMemFreeNodeParams { 
#line 2253
void *dptr; 
#line 2254
}; 
#endif
#line 2259 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2259
enum cudaGraphMemAttributeType { 
#line 2264
cudaGraphMemAttrUsedMemCurrent, 
#line 2271
cudaGraphMemAttrUsedMemHigh, 
#line 2278
cudaGraphMemAttrReservedMemCurrent, 
#line 2285
cudaGraphMemAttrReservedMemHigh
#line 2286
}; 
#endif
#line 2291 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2291
enum cudaMemcpyFlags { 
#line 2292
cudaMemcpyFlagDefault, 
#line 2297
cudaMemcpyFlagPreferOverlapWithCompute
#line 2298
}; 
#endif
#line 2300 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2300
enum cudaMemcpySrcAccessOrder { 
#line 2304
cudaMemcpySrcAccessOrderInvalid, 
#line 2309
cudaMemcpySrcAccessOrderStream, 
#line 2320
cudaMemcpySrcAccessOrderDuringApiCall, 
#line 2329
cudaMemcpySrcAccessOrderAny, 
#line 2331
cudaMemcpySrcAccessOrderMax = 2147483647
#line 2332
}; 
#endif
#line 2337 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2337
struct cudaMemcpyAttributes { 
#line 2338
cudaMemcpySrcAccessOrder srcAccessOrder; 
#line 2339
cudaMemLocation srcLocHint; 
#line 2340
cudaMemLocation dstLocHint; 
#line 2341
unsigned flags; 
#line 2342
}; 
#endif
#line 2347 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2347
enum cudaMemcpy3DOperandType { 
#line 2348
cudaMemcpyOperandTypePointer = 1, 
#line 2349
cudaMemcpyOperandTypeArray, 
#line 2350
cudaMemcpyOperandTypeMax = 2147483647
#line 2351
}; 
#endif
#line 2356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2356
struct cudaOffset3D { 
#line 2357
size_t x; 
#line 2358
size_t y; 
#line 2359
size_t z; 
#line 2360
}; 
#endif
#line 2365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2365
struct cudaMemcpy3DOperand { 
#line 2366
cudaMemcpy3DOperandType type; 
#line 2367
union { 
#line 2371
struct { 
#line 2372
void *ptr; 
#line 2373
size_t rowLength; 
#line 2374
size_t layerHeight; 
#line 2375
cudaMemLocation locHint; 
#line 2376
} ptr; 
#line 2381
struct { 
#line 2382
cudaArray_t array; 
#line 2383
cudaOffset3D offset; 
#line 2384
} array; 
#line 2385
} op; 
#line 2386
}; 
#endif
#line 2388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2388
struct cudaMemcpy3DBatchOp { 
#line 2389
cudaMemcpy3DOperand src; 
#line 2390
cudaMemcpy3DOperand dst; 
#line 2391
cudaExtent extent; 
#line 2392
cudaMemcpySrcAccessOrder srcAccessOrder; 
#line 2393
unsigned flags; 
#line 2394
}; 
#endif
#line 2400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2400
enum cudaDeviceP2PAttr { 
#line 2401
cudaDevP2PAttrPerformanceRank = 1, 
#line 2402
cudaDevP2PAttrAccessSupported, 
#line 2403
cudaDevP2PAttrNativeAtomicSupported, 
#line 2404
cudaDevP2PAttrCudaArrayAccessSupported
#line 2405
}; 
#endif
#line 2412 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2412
struct CUuuid_st { 
#line 2413
char bytes[16]; 
#line 2414
}; 
#endif
#line 2415 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef CUuuid_st 
#line 2415
CUuuid; 
#endif
#line 2417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef CUuuid_st 
#line 2417
cudaUUID_t; 
#endif
#line 2422 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2422
struct cudaDeviceProp { 
#line 2424
char name[256]; 
#line 2425
cudaUUID_t uuid; 
#line 2426
char luid[8]; 
#line 2427
unsigned luidDeviceNodeMask; 
#line 2428
size_t totalGlobalMem; 
#line 2429
size_t sharedMemPerBlock; 
#line 2430
int regsPerBlock; 
#line 2431
int warpSize; 
#line 2432
size_t memPitch; 
#line 2433
int maxThreadsPerBlock; 
#line 2434
int maxThreadsDim[3]; 
#line 2435
int maxGridSize[3]; 
#line 2436
int clockRate; 
#line 2437
size_t totalConstMem; 
#line 2438
int major; 
#line 2439
int minor; 
#line 2440
size_t textureAlignment; 
#line 2441
size_t texturePitchAlignment; 
#line 2442
int deviceOverlap; 
#line 2443
int multiProcessorCount; 
#line 2444
int kernelExecTimeoutEnabled; 
#line 2445
int integrated; 
#line 2446
int canMapHostMemory; 
#line 2447
int computeMode; 
#line 2448
int maxTexture1D; 
#line 2449
int maxTexture1DMipmap; 
#line 2450
int maxTexture1DLinear; 
#line 2451
int maxTexture2D[2]; 
#line 2452
int maxTexture2DMipmap[2]; 
#line 2453
int maxTexture2DLinear[3]; 
#line 2454
int maxTexture2DGather[2]; 
#line 2455
int maxTexture3D[3]; 
#line 2456
int maxTexture3DAlt[3]; 
#line 2457
int maxTextureCubemap; 
#line 2458
int maxTexture1DLayered[2]; 
#line 2459
int maxTexture2DLayered[3]; 
#line 2460
int maxTextureCubemapLayered[2]; 
#line 2461
int maxSurface1D; 
#line 2462
int maxSurface2D[2]; 
#line 2463
int maxSurface3D[3]; 
#line 2464
int maxSurface1DLayered[2]; 
#line 2465
int maxSurface2DLayered[3]; 
#line 2466
int maxSurfaceCubemap; 
#line 2467
int maxSurfaceCubemapLayered[2]; 
#line 2468
size_t surfaceAlignment; 
#line 2469
int concurrentKernels; 
#line 2470
int ECCEnabled; 
#line 2471
int pciBusID; 
#line 2472
int pciDeviceID; 
#line 2473
int pciDomainID; 
#line 2474
int tccDriver; 
#line 2475
int asyncEngineCount; 
#line 2476
int unifiedAddressing; 
#line 2477
int memoryClockRate; 
#line 2478
int memoryBusWidth; 
#line 2479
int l2CacheSize; 
#line 2480
int persistingL2CacheMaxSize; 
#line 2481
int maxThreadsPerMultiProcessor; 
#line 2482
int streamPrioritiesSupported; 
#line 2483
int globalL1CacheSupported; 
#line 2484
int localL1CacheSupported; 
#line 2485
size_t sharedMemPerMultiprocessor; 
#line 2486
int regsPerMultiprocessor; 
#line 2487
int managedMemory; 
#line 2488
int isMultiGpuBoard; 
#line 2489
int multiGpuBoardGroupID; 
#line 2490
int hostNativeAtomicSupported; 
#line 2491
int singleToDoublePrecisionPerfRatio; 
#line 2492
int pageableMemoryAccess; 
#line 2493
int concurrentManagedAccess; 
#line 2494
int computePreemptionSupported; 
#line 2495
int canUseHostPointerForRegisteredMem; 
#line 2496
int cooperativeLaunch; 
#line 2497
int cooperativeMultiDeviceLaunch; 
#line 2498
size_t sharedMemPerBlockOptin; 
#line 2499
int pageableMemoryAccessUsesHostPageTables; 
#line 2500
int directManagedMemAccessFromHost; 
#line 2501
int maxBlocksPerMultiProcessor; 
#line 2502
int accessPolicyMaxWindowSize; 
#line 2503
size_t reservedSharedMemPerBlock; 
#line 2504
int hostRegisterSupported; 
#line 2505
int sparseCudaArraySupported; 
#line 2506
int hostRegisterReadOnlySupported; 
#line 2507
int timelineSemaphoreInteropSupported; 
#line 2508
int memoryPoolsSupported; 
#line 2509
int gpuDirectRDMASupported; 
#line 2510
unsigned gpuDirectRDMAFlushWritesOptions; 
#line 2511
int gpuDirectRDMAWritesOrdering; 
#line 2512
unsigned memoryPoolSupportedHandleTypes; 
#line 2513
int deferredMappingCudaArraySupported; 
#line 2514
int ipcEventSupported; 
#line 2515
int clusterLaunch; 
#line 2516
int unifiedFunctionPointers; 
#line 2517
int reserved[63]; 
#line 2518
}; 
#endif
#line 2531 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 2528
struct cudaIpcEventHandle_st { 
#line 2530
char reserved[64]; 
#line 2531
} cudaIpcEventHandle_t; 
#endif
#line 2539 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 2536
struct cudaIpcMemHandle_st { 
#line 2538
char reserved[64]; 
#line 2539
} cudaIpcMemHandle_t; 
#endif
#line 2547 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 2544
struct cudaMemFabricHandle_st { 
#line 2546
char reserved[64]; 
#line 2547
} cudaMemFabricHandle_t; 
#endif
#line 2552 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2552
enum cudaExternalMemoryHandleType { 
#line 2556
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
#line 2560
cudaExternalMemoryHandleTypeOpaqueWin32, 
#line 2564
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
#line 2568
cudaExternalMemoryHandleTypeD3D12Heap, 
#line 2572
cudaExternalMemoryHandleTypeD3D12Resource, 
#line 2576
cudaExternalMemoryHandleTypeD3D11Resource, 
#line 2580
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
#line 2584
cudaExternalMemoryHandleTypeNvSciBuf
#line 2585
}; 
#endif
#line 2627 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2627
struct cudaExternalMemoryHandleDesc { 
#line 2631
cudaExternalMemoryHandleType type; 
#line 2632
union { 
#line 2638
int fd; 
#line 2654
struct { 
#line 2658
void *handle; 
#line 2663
const void *name; 
#line 2664
} win32; 
#line 2669
const void *nvSciBufObject; 
#line 2670
} handle; 
#line 2674
unsigned __int64 size; 
#line 2678
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 2679
}; 
#endif
#line 2684 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2684
struct cudaExternalMemoryBufferDesc { 
#line 2688
unsigned __int64 offset; 
#line 2692
unsigned __int64 size; 
#line 2696
unsigned flags; 
#line 2697
}; 
#endif
#line 2702 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2702
struct cudaExternalMemoryMipmappedArrayDesc { 
#line 2707
unsigned __int64 offset; 
#line 2711
cudaChannelFormatDesc formatDesc; 
#line 2715
cudaExtent extent; 
#line 2720
unsigned flags; 
#line 2724
unsigned numLevels; 
#line 2725
}; 
#endif
#line 2730 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2730
enum cudaExternalSemaphoreHandleType { 
#line 2734
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
#line 2738
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
#line 2742
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
#line 2746
cudaExternalSemaphoreHandleTypeD3D12Fence, 
#line 2750
cudaExternalSemaphoreHandleTypeD3D11Fence, 
#line 2754
cudaExternalSemaphoreHandleTypeNvSciSync, 
#line 2758
cudaExternalSemaphoreHandleTypeKeyedMutex, 
#line 2762
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, 
#line 2766
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, 
#line 2770
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
#line 2771
}; 
#endif
#line 2776 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2776
struct cudaExternalSemaphoreHandleDesc { 
#line 2780
cudaExternalSemaphoreHandleType type; 
#line 2781
union { 
#line 2788
int fd; 
#line 2804
struct { 
#line 2808
void *handle; 
#line 2813
const void *name; 
#line 2814
} win32; 
#line 2818
const void *nvSciSyncObj; 
#line 2819
} handle; 
#line 2823
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 2824
}; 
#endif
#line 2829 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2829
struct cudaExternalSemaphoreSignalParams_v1 { 
#line 2830
struct { 
#line 2834
struct { 
#line 2838
unsigned __int64 value; 
#line 2839
} fence; 
#line 2840
union { 
#line 2845
void *fence; 
#line 2846
unsigned __int64 reserved; 
#line 2847
} nvSciSync; 
#line 2851
struct { 
#line 2855
unsigned __int64 key; 
#line 2856
} keyedMutex; 
#line 2857
} params; 
#line 2868
unsigned flags; 
#line 2869
}; 
#endif
#line 2874 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2874
struct cudaExternalSemaphoreWaitParams_v1 { 
#line 2875
struct { 
#line 2879
struct { 
#line 2883
unsigned __int64 value; 
#line 2884
} fence; 
#line 2885
union { 
#line 2890
void *fence; 
#line 2891
unsigned __int64 reserved; 
#line 2892
} nvSciSync; 
#line 2896
struct { 
#line 2900
unsigned __int64 key; 
#line 2904
unsigned timeoutMs; 
#line 2905
} keyedMutex; 
#line 2906
} params; 
#line 2917
unsigned flags; 
#line 2918
}; 
#endif
#line 2923 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2923
struct cudaExternalSemaphoreSignalParams { 
#line 2924
struct { 
#line 2928
struct { 
#line 2932
unsigned __int64 value; 
#line 2933
} fence; 
#line 2934
union { 
#line 2939
void *fence; 
#line 2940
unsigned __int64 reserved; 
#line 2941
} nvSciSync; 
#line 2945
struct { 
#line 2949
unsigned __int64 key; 
#line 2950
} keyedMutex; 
#line 2951
unsigned reserved[12]; 
#line 2952
} params; 
#line 2963
unsigned flags; 
#line 2964
unsigned reserved[16]; 
#line 2965
}; 
#endif
#line 2970 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 2970
struct cudaExternalSemaphoreWaitParams { 
#line 2971
struct { 
#line 2975
struct { 
#line 2979
unsigned __int64 value; 
#line 2980
} fence; 
#line 2981
union { 
#line 2986
void *fence; 
#line 2987
unsigned __int64 reserved; 
#line 2988
} nvSciSync; 
#line 2992
struct { 
#line 2996
unsigned __int64 key; 
#line 3000
unsigned timeoutMs; 
#line 3001
} keyedMutex; 
#line 3002
unsigned reserved[10]; 
#line 3003
} params; 
#line 3014
unsigned flags; 
#line 3015
unsigned reserved[16]; 
#line 3016
}; 
#endif
#line 3027 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef cudaError 
#line 3027
cudaError_t; 
#endif
#line 3032 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUstream_st *
#line 3032
cudaStream_t; 
#endif
#line 3037 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUevent_st *
#line 3037
cudaEvent_t; 
#endif
#line 3042 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef cudaGraphicsResource *
#line 3042
cudaGraphicsResource_t; 
#endif
#line 3047 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
#line 3047
cudaExternalMemory_t; 
#endif
#line 3052 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
#line 3052
cudaExternalSemaphore_t; 
#endif
#line 3057 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUgraph_st *
#line 3057
cudaGraph_t; 
#endif
#line 3062 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUgraphNode_st *
#line 3062
cudaGraphNode_t; 
#endif
#line 3067 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUuserObject_st *
#line 3067
cudaUserObject_t; 
#endif
#line 3072 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef unsigned __int64 
#line 3072
cudaGraphConditionalHandle; 
#endif
#line 3077 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUfunc_st *
#line 3077
cudaFunction_t; 
#endif
#line 3082 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUkern_st *
#line 3082
cudaKernel_t; 
#endif
#line 3087 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3087
enum cudaJitOption { 
#line 3094
cudaJitMaxRegisters, 
#line 3108
cudaJitThreadsPerBlock, 
#line 3116
cudaJitWallTime, 
#line 3125
cudaJitInfoLogBuffer, 
#line 3134
cudaJitInfoLogBufferSizeBytes, 
#line 3143
cudaJitErrorLogBuffer, 
#line 3152
cudaJitErrorLogBufferSizeBytes, 
#line 3160
cudaJitOptimizationLevel, 
#line 3168
cudaJitFallbackStrategy = 10, 
#line 3176
cudaJitGenerateDebugInfo, 
#line 3183
cudaJitLogVerbose, 
#line 3190
cudaJitGenerateLineInfo, 
#line 3198
cudaJitCacheMode, 
#line 3205
cudaJitPositionIndependentCode = 30, 
#line 3218
cudaJitMinCtaPerSm, 
#line 3231
cudaJitMaxThreadsPerBlock, 
#line 3241
cudaJitOverrideDirectiveValues
#line 3242
}; 
#endif
#line 3248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3248
enum cudaLibraryOption { 
#line 3250
cudaLibraryHostUniversalFunctionAndDataTable, 
#line 3261
cudaLibraryBinaryIsPreserved
#line 3262
}; 
#endif
#line 3264 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3264
struct cudalibraryHostUniversalFunctionAndDataTable { 
#line 3266
void *functionTable; 
#line 3267
size_t functionWindowSize; 
#line 3268
void *dataTable; 
#line 3269
size_t dataWindowSize; 
#line 3270
}; 
#endif
#line 3275 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3275
enum cudaJit_CacheMode { 
#line 3277
cudaJitCacheOptionNone, 
#line 3278
cudaJitCacheOptionCG, 
#line 3279
cudaJitCacheOptionCA
#line 3280
}; 
#endif
#line 3285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3285
enum cudaJit_Fallback { 
#line 3287
cudaPreferPtx, 
#line 3289
cudaPreferBinary
#line 3290
}; 
#endif
#line 3295 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUlib_st *
#line 3295
cudaLibrary_t; 
#endif
#line 3300 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef struct CUmemPoolHandle_st *
#line 3300
cudaMemPool_t; 
#endif
#line 3305 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3305
enum cudaCGScope { 
#line 3306
cudaCGScopeInvalid, 
#line 3307
cudaCGScopeGrid, 
#line 3308
cudaCGScopeMultiGrid
#line 3309
}; 
#endif
#line 3314 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3314
struct cudaLaunchParams { 
#line 3316
void *func; 
#line 3317
dim3 gridDim; 
#line 3318
dim3 blockDim; 
#line 3319
void **args; 
#line 3320
size_t sharedMem; 
#line 3321
cudaStream_t stream; 
#line 3322
}; 
#endif
#line 3327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3327
struct cudaKernelNodeParams { 
#line 3328
void *func; 
#line 3329
dim3 gridDim; 
#line 3330
dim3 blockDim; 
#line 3331
unsigned sharedMemBytes; 
#line 3332
void **kernelParams; 
#line 3333
void **extra; 
#line 3334
}; 
#endif
#line 3339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3339
struct cudaKernelNodeParamsV2 { 
#line 3340
void *func; 
#line 3346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
uint3 gridDim; 
#line 3347
uint3 blockDim; 
#line 3349 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
unsigned sharedMemBytes; 
#line 3350
void **kernelParams; 
#line 3351
void **extra; 
#line 3352
}; 
#endif
#line 3357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3357
struct cudaExternalSemaphoreSignalNodeParams { 
#line 3358
cudaExternalSemaphore_t *extSemArray; 
#line 3359
const cudaExternalSemaphoreSignalParams *paramsArray; 
#line 3360
unsigned numExtSems; 
#line 3361
}; 
#endif
#line 3366 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3366
struct cudaExternalSemaphoreSignalNodeParamsV2 { 
#line 3367
cudaExternalSemaphore_t *extSemArray; 
#line 3368
const cudaExternalSemaphoreSignalParams *paramsArray; 
#line 3369
unsigned numExtSems; 
#line 3370
}; 
#endif
#line 3375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3375
struct cudaExternalSemaphoreWaitNodeParams { 
#line 3376
cudaExternalSemaphore_t *extSemArray; 
#line 3377
const cudaExternalSemaphoreWaitParams *paramsArray; 
#line 3378
unsigned numExtSems; 
#line 3379
}; 
#endif
#line 3384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3384
struct cudaExternalSemaphoreWaitNodeParamsV2 { 
#line 3385
cudaExternalSemaphore_t *extSemArray; 
#line 3386
const cudaExternalSemaphoreWaitParams *paramsArray; 
#line 3387
unsigned numExtSems; 
#line 3388
}; 
#endif
#line 3390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3390
enum cudaGraphConditionalHandleFlags { 
#line 3391
cudaGraphCondAssignDefault = 1
#line 3392
}; 
#endif
#line 3397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3397
enum cudaGraphConditionalNodeType { 
#line 3398
cudaGraphCondTypeIf, 
#line 3399
cudaGraphCondTypeWhile, 
#line 3400
cudaGraphCondTypeSwitch
#line 3401
}; 
#endif
#line 3406 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3406
struct cudaConditionalNodeParams { 
#line 3407
cudaGraphConditionalHandle handle; 
#line 3410
cudaGraphConditionalNodeType type; 
#line 3411
unsigned size; 
#line 3413
cudaGraph_t *phGraph_out; 
#line 3432
}; 
#endif
#line 3437 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3437
enum cudaGraphNodeType { 
#line 3438
cudaGraphNodeTypeKernel, 
#line 3439
cudaGraphNodeTypeMemcpy, 
#line 3440
cudaGraphNodeTypeMemset, 
#line 3441
cudaGraphNodeTypeHost, 
#line 3442
cudaGraphNodeTypeGraph, 
#line 3443
cudaGraphNodeTypeEmpty, 
#line 3444
cudaGraphNodeTypeWaitEvent, 
#line 3445
cudaGraphNodeTypeEventRecord, 
#line 3446
cudaGraphNodeTypeExtSemaphoreSignal, 
#line 3447
cudaGraphNodeTypeExtSemaphoreWait, 
#line 3448
cudaGraphNodeTypeMemAlloc, 
#line 3449
cudaGraphNodeTypeMemFree, 
#line 3450
cudaGraphNodeTypeConditional = 13, 
#line 3467
cudaGraphNodeTypeCount
#line 3468
}; 
#endif
#line 3473 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3473
struct cudaChildGraphNodeParams { 
#line 3474
cudaGraph_t graph; 
#line 3476
}; 
#endif
#line 3481 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3481
struct cudaEventRecordNodeParams { 
#line 3482
cudaEvent_t event; 
#line 3483
}; 
#endif
#line 3488 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3488
struct cudaEventWaitNodeParams { 
#line 3489
cudaEvent_t event; 
#line 3490
}; 
#endif
#line 3495 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3495
struct cudaGraphNodeParams { 
#line 3496
cudaGraphNodeType type; 
#line 3497
int reserved0[3]; 
#line 3499
union { 
#line 3500
__int64 reserved1[29]; 
#line 3501
cudaKernelNodeParamsV2 kernel; 
#line 3502
cudaMemcpyNodeParams memcpy; 
#line 3503
cudaMemsetParamsV2 memset; 
#line 3504
cudaHostNodeParamsV2 host; 
#line 3505
cudaChildGraphNodeParams graph; 
#line 3506
cudaEventWaitNodeParams eventWait; 
#line 3507
cudaEventRecordNodeParams eventRecord; 
#line 3508
cudaExternalSemaphoreSignalNodeParamsV2 extSemSignal; 
#line 3509
cudaExternalSemaphoreWaitNodeParamsV2 extSemWait; 
#line 3510
cudaMemAllocNodeParamsV2 alloc; 
#line 3511
cudaMemFreeNodeParams free; 
#line 3512
cudaConditionalNodeParams conditional; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 3513
}; 
#line 3515
__int64 reserved2; 
#line 3516
}; 
#endif
#line 3528 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3521
enum cudaGraphDependencyType_enum { 
#line 3522
cudaGraphDependencyTypeDefault, 
#line 3523
cudaGraphDependencyTypeProgrammatic
#line 3528
} cudaGraphDependencyType; 
#endif
#line 3558 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3535
struct cudaGraphEdgeData_st { 
#line 3536
unsigned char from_port; 
#line 3546
unsigned char to_port; 
#line 3553
unsigned char type; 
#line 3556
unsigned char reserved[5]; 
#line 3558
} cudaGraphEdgeData; 
#endif
#line 3579 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
#line 3584
#if 0
#line 3584
enum cudaGraphExecUpdateResult { 
#line 3585
cudaGraphExecUpdateSuccess, 
#line 3586
cudaGraphExecUpdateError, 
#line 3587
cudaGraphExecUpdateErrorTopologyChanged, 
#line 3588
cudaGraphExecUpdateErrorNodeTypeChanged, 
#line 3589
cudaGraphExecUpdateErrorFunctionChanged, 
#line 3590
cudaGraphExecUpdateErrorParametersChanged, 
#line 3591
cudaGraphExecUpdateErrorNotSupported, 
#line 3592
cudaGraphExecUpdateErrorUnsupportedFunctionChange, 
#line 3593
cudaGraphExecUpdateErrorAttributesChanged
#line 3594
}; 
#endif
#line 3606 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3599
enum cudaGraphInstantiateResult { 
#line 3600
cudaGraphInstantiateSuccess, 
#line 3601
cudaGraphInstantiateError, 
#line 3602
cudaGraphInstantiateInvalidStructure, 
#line 3603
cudaGraphInstantiateNodeOperationNotSupported, 
#line 3604
cudaGraphInstantiateMultipleDevicesNotSupported, 
#line 3605
cudaGraphInstantiateConditionalHandleUnused
#line 3606
} cudaGraphInstantiateResult; 
#endif
#line 3617 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3611
struct cudaGraphInstantiateParams_st { 
#line 3613
unsigned __int64 flags; 
#line 3614
cudaStream_t uploadStream; 
#line 3615
cudaGraphNode_t errNode_out; 
#line 3616
cudaGraphInstantiateResult result_out; 
#line 3617
} cudaGraphInstantiateParams; 
#endif
#line 3639 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3622
struct cudaGraphExecUpdateResultInfo_st { 
#line 3626
cudaGraphExecUpdateResult result; 
#line 3633
cudaGraphNode_t errorNode; 
#line 3638
cudaGraphNode_t errorFromNode; 
#line 3639
} cudaGraphExecUpdateResultInfo; 
#endif
#line 3644 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
typedef struct CUgraphDeviceUpdatableNode_st *cudaGraphDeviceNode_t; 
#line 3649
#if 0
#line 3649
enum cudaGraphKernelNodeField { 
#line 3651
cudaGraphKernelNodeFieldInvalid, 
#line 3652
cudaGraphKernelNodeFieldGridDim, 
#line 3653
cudaGraphKernelNodeFieldParam, 
#line 3654
cudaGraphKernelNodeFieldEnabled
#line 3655
}; 
#endif
#line 3660 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3660
struct cudaGraphKernelNodeUpdate { 
#line 3661
cudaGraphDeviceNode_t node; 
#line 3662
cudaGraphKernelNodeField field; 
#line 3663
union { 
#line 3668 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
uint3 gridDim; 
#line 3670 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
struct { 
#line 3671
const void *pValue; 
#line 3672
size_t offset; 
#line 3673
size_t size; 
#line 3674
} param; 
#line 3675
unsigned isEnabled; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 3676
} updateData; 
#line 3677
}; 
#endif
#line 3683 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3683
enum cudaGetDriverEntryPointFlags { 
#line 3684
cudaEnableDefault, 
#line 3685
cudaEnableLegacyStream, 
#line 3686
cudaEnablePerThreadDefaultStream
#line 3687
}; 
#endif
#line 3692 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3692
enum cudaDriverEntryPointQueryResult { 
#line 3693
cudaDriverEntryPointSuccess, 
#line 3694
cudaDriverEntryPointSymbolNotFound, 
#line 3695
cudaDriverEntryPointVersionNotSufficent
#line 3696
}; 
#endif
#line 3701 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3701
enum cudaGraphDebugDotFlags { 
#line 3702
cudaGraphDebugDotFlagsVerbose = (1 << 0), 
#line 3703
cudaGraphDebugDotFlagsKernelNodeParams = (1 << 2), 
#line 3704
cudaGraphDebugDotFlagsMemcpyNodeParams = (1 << 3), 
#line 3705
cudaGraphDebugDotFlagsMemsetNodeParams = (1 << 4), 
#line 3706
cudaGraphDebugDotFlagsHostNodeParams = (1 << 5), 
#line 3707
cudaGraphDebugDotFlagsEventNodeParams = (1 << 6), 
#line 3708
cudaGraphDebugDotFlagsExtSemasSignalNodeParams = (1 << 7), 
#line 3709
cudaGraphDebugDotFlagsExtSemasWaitNodeParams = (1 << 8), 
#line 3710
cudaGraphDebugDotFlagsKernelNodeAttributes = (1 << 9), 
#line 3711
cudaGraphDebugDotFlagsHandles = (1 << 10), 
#line 3712
cudaGraphDebugDotFlagsConditionalNodeParams = (1 << 15)
#line 3713
}; 
#endif
#line 3718 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 3718
enum cudaGraphInstantiateFlags { 
#line 3719
cudaGraphInstantiateFlagAutoFreeOnLaunch = 1, 
#line 3720
cudaGraphInstantiateFlagUpload, 
#line 3723
cudaGraphInstantiateFlagDeviceLaunch = 4, 
#line 3726
cudaGraphInstantiateFlagUseNodePriority = 8
#line 3728
}; 
#endif
#line 3749 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3746
enum cudaLaunchMemSyncDomain { 
#line 3747
cudaLaunchMemSyncDomainDefault, 
#line 3748
cudaLaunchMemSyncDomainRemote
#line 3749
} cudaLaunchMemSyncDomain; 
#endif
#line 3765 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3762
struct cudaLaunchMemSyncDomainMap_st { 
#line 3763
unsigned char default_; 
#line 3764
unsigned char remote; 
#line 3765
} cudaLaunchMemSyncDomainMap; 
#endif
#line 3917 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3770
enum cudaLaunchAttributeID { 
#line 3771
cudaLaunchAttributeIgnore, 
#line 3772
cudaLaunchAttributeAccessPolicyWindow, 
#line 3774
cudaLaunchAttributeCooperative, 
#line 3776
cudaLaunchAttributeSynchronizationPolicy, 
#line 3777
cudaLaunchAttributeClusterDimension, 
#line 3779
cudaLaunchAttributeClusterSchedulingPolicyPreference, 
#line 3781
cudaLaunchAttributeProgrammaticStreamSerialization, 
#line 3792
cudaLaunchAttributeProgrammaticEvent, 
#line 3818
cudaLaunchAttributePriority, 
#line 3820
cudaLaunchAttributeMemSyncDomainMap, 
#line 3822
cudaLaunchAttributeMemSyncDomain, 
#line 3824
cudaLaunchAttributePreferredClusterDimension, 
#line 3860
cudaLaunchAttributeLaunchCompletionEvent, 
#line 3882
cudaLaunchAttributeDeviceUpdatableKernelNode, 
#line 3910
cudaLaunchAttributePreferredSharedMemoryCarveout
#line 3917
} cudaLaunchAttributeID; 
#endif
#line 4013 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 3922
union cudaLaunchAttributeValue { 
#line 3923
char pad[64]; 
#line 3924
cudaAccessPolicyWindow accessPolicyWindow; 
#line 3925
int cooperative; 
#line 3927
cudaSynchronizationPolicy syncPolicy; 
#line 3941
struct { 
#line 3942
unsigned x; 
#line 3943
unsigned y; 
#line 3944
unsigned z; 
#line 3945
} clusterDim; 
#line 3946
cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; 
#line 3949
int programmaticStreamSerializationAllowed; 
#line 3960
struct { 
#line 3961
cudaEvent_t event; 
#line 3962
int flags; 
#line 3963
int triggerAtBlockStart; 
#line 3964
} programmaticEvent; 
#line 3965
int priority; 
#line 3966
cudaLaunchMemSyncDomainMap memSyncDomainMap; 
#line 3969
cudaLaunchMemSyncDomain memSyncDomain; 
#line 3984
struct { 
#line 3985
unsigned x; 
#line 3986
unsigned y; 
#line 3987
unsigned z; 
#line 3988
} preferredClusterDim; 
#line 3997
struct { 
#line 3998
cudaEvent_t event; 
#line 3999
int flags; 
#line 4000
} launchCompletionEvent; 
#line 4008
struct { 
#line 4009
int deviceUpdatable; 
#line 4010
cudaGraphDeviceNode_t devNode; 
#line 4011
} deviceUpdatableKernelNode; 
#line 4012
unsigned sharedMemCarveout; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 4013
} cudaLaunchAttributeValue; 
#endif
#line 4022 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 4018
struct cudaLaunchAttribute_st { 
#line 4019
cudaLaunchAttributeID id; 
#line 4020
char pad[(8) - sizeof(cudaLaunchAttributeID)]; 
#line 4021
cudaLaunchAttributeValue val; 
#line 4022
} cudaLaunchAttribute; 
#endif
#line 4034 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 4027
struct cudaLaunchConfig_st { 
#line 4028
dim3 gridDim; 
#line 4029
dim3 blockDim; 
#line 4030
size_t dynamicSmemBytes; 
#line 4031
cudaStream_t stream; 
#line 4032
cudaLaunchAttribute *attrs; 
#line 4033
unsigned numAttrs; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 4034
} cudaLaunchConfig_t; 
#endif
#line 4061 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
#line 4061
enum cudaDeviceNumaConfig { 
#line 4062
cudaDeviceNumaConfigNone, 
#line 4063
cudaDeviceNumaConfigNumaNode
#line 4064
}; 
#endif
#line 4069 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
typedef struct cudaAsyncCallbackEntry *cudaAsyncCallbackHandle_t; 
#line 4071
struct cudaAsyncCallbackEntry; 
#line 4078
#if 0
typedef 
#line 4076
enum cudaAsyncNotificationType_enum { 
#line 4077
cudaAsyncNotificationTypeOverBudget = 1
#line 4078
} cudaAsyncNotificationType; 
#endif
#line 4091 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
#if 0
typedef 
#line 4083
struct cudaAsyncNotificationInfo { 
#line 4085
cudaAsyncNotificationType type; 
#line 4086
union { 
#line 4087
struct { 
#line 4088
unsigned __int64 bytesOverBudget; 
#line 4089
} overBudget; 
#line 4090
} info; 
#line 4091
} cudaAsyncNotificationInfo_t; 
#endif
#line 4093 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_types.h"
typedef void (*cudaAsyncCallback)(cudaAsyncNotificationInfo_t *, void *, cudaAsyncCallbackHandle_t); 
#line 86 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_types.h"
#if 0
#line 86
enum cudaSurfaceBoundaryMode { 
#line 88
cudaBoundaryModeZero, 
#line 89
cudaBoundaryModeClamp, 
#line 90
cudaBoundaryModeTrap
#line 91
}; 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_types.h"
#if 0
#line 96
enum cudaSurfaceFormatMode { 
#line 98
cudaFormatModeForced, 
#line 99
cudaFormatModeAuto
#line 100
}; 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_types.h"
#if 0
typedef unsigned __int64 
#line 105
cudaSurfaceObject_t; 
#endif
#line 86 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_types.h"
#if 0
#line 86
enum cudaTextureAddressMode { 
#line 88
cudaAddressModeWrap, 
#line 89
cudaAddressModeClamp, 
#line 90
cudaAddressModeMirror, 
#line 91
cudaAddressModeBorder
#line 92
}; 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_types.h"
#if 0
#line 97
enum cudaTextureFilterMode { 
#line 99
cudaFilterModePoint, 
#line 100
cudaFilterModeLinear
#line 101
}; 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_types.h"
#if 0
#line 106
enum cudaTextureReadMode { 
#line 108
cudaReadModeElementType, 
#line 109
cudaReadModeNormalizedFloat
#line 110
}; 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_types.h"
#if 0
#line 115
struct cudaTextureDesc { 
#line 120
cudaTextureAddressMode addressMode[3]; 
#line 124
cudaTextureFilterMode filterMode; 
#line 128
cudaTextureReadMode readMode; 
#line 132
int sRGB; 
#line 136
float borderColor[4]; 
#line 140
int normalizedCoords; 
#line 144
unsigned maxAnisotropy; 
#line 148
cudaTextureFilterMode mipmapFilterMode; 
#line 152
float mipmapLevelBias; 
#line 156
float minMipmapLevelClamp; 
#line 160
float maxMipmapLevelClamp; 
#line 164
int disableTrilinearOptimization; 
#line 168
int seamlessCubemap; 
#line 169
}; 
#endif
#line 174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_types.h"
#if 0
typedef unsigned __int64 
#line 174
cudaTextureObject_t; 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\library_types.h"
typedef 
#line 57
enum cudaDataType_t { 
#line 59
CUDA_R_16F = 2, 
#line 60
CUDA_C_16F = 6, 
#line 61
CUDA_R_16BF = 14, 
#line 62
CUDA_C_16BF, 
#line 63
CUDA_R_32F = 0, 
#line 64
CUDA_C_32F = 4, 
#line 65
CUDA_R_64F = 1, 
#line 66
CUDA_C_64F = 5, 
#line 67
CUDA_R_4I = 16, 
#line 68
CUDA_C_4I, 
#line 69
CUDA_R_4U, 
#line 70
CUDA_C_4U, 
#line 71
CUDA_R_8I = 3, 
#line 72
CUDA_C_8I = 7, 
#line 73
CUDA_R_8U, 
#line 74
CUDA_C_8U, 
#line 75
CUDA_R_16I = 20, 
#line 76
CUDA_C_16I, 
#line 77
CUDA_R_16U, 
#line 78
CUDA_C_16U, 
#line 79
CUDA_R_32I = 10, 
#line 80
CUDA_C_32I, 
#line 81
CUDA_R_32U, 
#line 82
CUDA_C_32U, 
#line 83
CUDA_R_64I = 24, 
#line 84
CUDA_C_64I, 
#line 85
CUDA_R_64U, 
#line 86
CUDA_C_64U, 
#line 87
CUDA_R_8F_E4M3, 
#line 88
CUDA_R_8F_UE4M3 = CUDA_R_8F_E4M3, 
#line 89
CUDA_R_8F_E5M2, 
#line 90
CUDA_R_8F_UE8M0, 
#line 91
CUDA_R_6F_E2M3, 
#line 92
CUDA_R_6F_E3M2, 
#line 93
CUDA_R_4F_E2M1
#line 94
} cudaDataType; 
#line 102
typedef 
#line 97
enum libraryPropertyType_t { 
#line 99
MAJOR_VERSION, 
#line 100
MINOR_VERSION, 
#line 101
PATCH_LEVEL
#line 102
} libraryPropertyType; 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_malloc.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 58 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_malloc.h"
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 59
_calloc_base(size_t _Count, size_t _Size); 
#line 65
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 66
calloc(size_t _Count, size_t _Size); 
#line 72
int __cdecl _callnewh(size_t _Size); 
#line 77
__declspec(allocator) void *__cdecl 
#line 78
_expand(void * _Block, size_t _Size); 
#line 84
void __cdecl _free_base(void * _Block); 
#line 89
void __cdecl free(void * _Block); 
#line 94
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 95
_malloc_base(size_t _Size); 
#line 100
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 101
malloc(size_t _Size); 
#line 107
size_t __cdecl _msize_base(void * _Block) noexcept; 
#line 113
size_t __cdecl _msize(void * _Block); 
#line 118
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 119
_realloc_base(void * _Block, size_t _Size); 
#line 125
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 126
realloc(void * _Block, size_t _Size); 
#line 132
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 133
_recalloc_base(void * _Block, size_t _Count, size_t _Size); 
#line 140
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 141
_recalloc(void * _Block, size_t _Count, size_t _Size); 
#line 148
void __cdecl _aligned_free(void * _Block); 
#line 153
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 154
_aligned_malloc(size_t _Size, size_t _Alignment); 
#line 160
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 161
_aligned_offset_malloc(size_t _Size, size_t _Alignment, size_t _Offset); 
#line 169
size_t __cdecl _aligned_msize(void * _Block, size_t _Alignment, size_t _Offset); 
#line 176
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 177
_aligned_offset_realloc(void * _Block, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 185
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 186
_aligned_offset_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 195
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 196
_aligned_realloc(void * _Block, size_t _Size, size_t _Alignment); 
#line 203
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 204
_aligned_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment); 
#line 232 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_malloc.h"
}__pragma( pack ( pop )) 
#line 234
#pragma warning(pop)
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_search.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 20
__pragma( pack ( push, 8 )) extern "C" {
#line 23
typedef int (__cdecl *_CoreCrtSecureSearchSortCompareFunction)(void *, const void *, const void *); 
#line 24
typedef int (__cdecl *_CoreCrtNonSecureSearchSortCompareFunction)(const void *, const void *); 
#line 30
void *__cdecl bsearch_s(const void * _Key, const void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 39
void __cdecl qsort_s(void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 52 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_search.h"
void *__cdecl bsearch(const void * _Key, const void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 60
void __cdecl qsort(void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 68
void *__cdecl _lfind_s(const void * _Key, const void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 78
void *__cdecl _lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 87
void *__cdecl _lsearch_s(const void * _Key, void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 97
void *__cdecl _lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 195 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_search.h"
void *__cdecl lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 204
void *__cdecl lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 216 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_search.h"
}__pragma( pack ( pop )) 
#line 218
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 54
errno_t __cdecl _itow_s(int _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 61
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itow_s ( int _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 68 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _itow(int _Value, __wchar_t * _Buffer, int _Radix); 
#line 77 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ltow_s(long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 84
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltow_s ( long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 91 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ltow(long _Value, __wchar_t * _Buffer, int _Radix); 
#line 99 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ultow_s(unsigned long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 106
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultow_s ( unsigned long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 113 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ultow(unsigned long _Value, __wchar_t * _Buffer, int _Radix); 
#line 121 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
double __cdecl wcstod(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 127
double __cdecl _wcstod_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 134
long __cdecl wcstol(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 141
long __cdecl _wcstol_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 149
__int64 __cdecl wcstoll(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 156
__int64 __cdecl _wcstoll_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 164
unsigned long __cdecl wcstoul(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 171
unsigned long __cdecl _wcstoul_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 179
unsigned __int64 __cdecl wcstoull(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 186
unsigned __int64 __cdecl _wcstoull_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 194
long double __cdecl wcstold(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 200
long double __cdecl _wcstold_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 207
float __cdecl wcstof(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 213
float __cdecl _wcstof_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 220
double __cdecl _wtof(const __wchar_t * _String); 
#line 225
double __cdecl _wtof_l(const __wchar_t * _String, _locale_t _Locale); 
#line 231
int __cdecl _wtoi(const __wchar_t * _String); 
#line 236
int __cdecl _wtoi_l(const __wchar_t * _String, _locale_t _Locale); 
#line 242
long __cdecl _wtol(const __wchar_t * _String); 
#line 247
long __cdecl _wtol_l(const __wchar_t * _String, _locale_t _Locale); 
#line 253
__int64 __cdecl _wtoll(const __wchar_t * _String); 
#line 258
__int64 __cdecl _wtoll_l(const __wchar_t * _String, _locale_t _Locale); 
#line 264
errno_t __cdecl _i64tow_s(__int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 272
__wchar_t *__cdecl _i64tow(__int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 279
errno_t __cdecl _ui64tow_s(unsigned __int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 287
__wchar_t *__cdecl _ui64tow(unsigned __int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 294
__int64 __cdecl _wtoi64(const __wchar_t * _String); 
#line 299
__int64 __cdecl _wtoi64_l(const __wchar_t * _String, _locale_t _Locale); 
#line 305
__int64 __cdecl _wcstoi64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 312
__int64 __cdecl _wcstoi64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 320
unsigned __int64 __cdecl _wcstoui64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 327
unsigned __int64 __cdecl _wcstoui64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 339
__declspec(allocator) __wchar_t *__cdecl _wfullpath(__wchar_t * _Buffer, const __wchar_t * _Path, size_t _BufferCount); 
#line 348
errno_t __cdecl _wmakepath_s(__wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 357
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wmakepath_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _Drive, wchar_t const * _Dir, wchar_t const * _Filename, wchar_t const * _Ext ) throw ( ) { return _wmakepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 366 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wmakepath(__wchar_t * _Buffer, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 375 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wperror(const __wchar_t * _ErrorMessage); 
#line 380
void __cdecl _wsplitpath(const __wchar_t * _FullPath, __wchar_t * _Drive, __wchar_t * _Dir, __wchar_t * _Filename, __wchar_t * _Ext); 
#line 388
errno_t __cdecl _wsplitpath_s(const __wchar_t * _FullPath, __wchar_t * _Drive, size_t _DriveCount, __wchar_t * _Dir, size_t _DirCount, __wchar_t * _Filename, size_t _FilenameCount, __wchar_t * _Ext, size_t _ExtCount); 
#line 400
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _wsplitpath_s ( wchar_t const * _Path, wchar_t ( & _Drive ) [ _DriveSize ], wchar_t ( & _Dir ) [ _DirSize ], wchar_t ( & _Name ) [ _NameSize ], wchar_t ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _wsplitpath_s ( _Path, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 409 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _wdupenv_s(__wchar_t ** _Buffer, size_t * _BufferCount, const __wchar_t * _VarName); 
#line 418
__wchar_t *__cdecl _wgetenv(const __wchar_t * _VarName); 
#line 424
errno_t __cdecl _wgetenv_s(size_t * _RequiredCount, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _VarName); 
#line 431
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wgetenv_s ( size_t * _RequiredCount, wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _VarName ) throw ( ) { return _wgetenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 440 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wputenv(const __wchar_t * _EnvString); 
#line 445
errno_t __cdecl _wputenv_s(const __wchar_t * _Name, const __wchar_t * _Value); 
#line 450
errno_t __cdecl _wsearchenv_s(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _Buffer, size_t _BufferCount); 
#line 457
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wsearchenv_s ( wchar_t const * _Filename, wchar_t const * _VarName, wchar_t ( & _ResultPath ) [ _Size ] ) throw ( ) { return _wsearchenv_s ( _Filename, _VarName, _ResultPath, _Size ); }}
#line 464 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wsearchenv(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _ResultPath); 
#line 471 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wsystem(const __wchar_t * _Command); 
#line 479 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstdlib.h"
}__pragma( pack ( pop )) 
#line 481
#pragma warning(pop)
#line 18 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 22
__pragma( pack ( push, 8 )) extern "C" {
#line 38 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
void __cdecl _swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 56
__declspec(noreturn) void __cdecl exit(int _Code); 
#line 57
__declspec(noreturn) void __cdecl _exit(int _Code); 
#line 58
__declspec(noreturn) void __cdecl _Exit(int _Code); 
#line 59
__declspec(noreturn) void __cdecl quick_exit(int _Code); 
#line 60
__declspec(noreturn) void __cdecl abort(); 
#line 67 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
unsigned __cdecl _set_abort_behavior(unsigned _Flags, unsigned _Mask); 
#line 77
typedef int (__cdecl *_onexit_t)(void); 
#line 144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
int __cdecl atexit(void (__cdecl *)(void)); 
#line 145
_onexit_t __cdecl _onexit(_onexit_t _Func); 
#line 148 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
int __cdecl at_quick_exit(void (__cdecl *)(void)); 
#line 159
typedef void (__cdecl *_purecall_handler)(void); 
#line 162
typedef void (__cdecl *_invalid_parameter_handler)(const __wchar_t *, const __wchar_t *, const __wchar_t *, unsigned, uintptr_t); 
#line 171
_purecall_handler __cdecl _set_purecall_handler(_purecall_handler _Handler); 
#line 175
_purecall_handler __cdecl _get_purecall_handler(); 
#line 178
_invalid_parameter_handler __cdecl _set_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 182
_invalid_parameter_handler __cdecl _get_invalid_parameter_handler(); 
#line 184
_invalid_parameter_handler __cdecl _set_thread_local_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 188
_invalid_parameter_handler __cdecl _get_thread_local_invalid_parameter_handler(); 
#line 212 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
int __cdecl _set_error_mode(int _Mode); 
#line 217
int *__cdecl _errno(); 
#line 220
errno_t __cdecl _set_errno(int _Value); 
#line 221
errno_t __cdecl _get_errno(int * _Value); 
#line 223
unsigned long *__cdecl __doserrno(); 
#line 226
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 227
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 230
char **__cdecl __sys_errlist(); 
#line 233
int *__cdecl __sys_nerr(); 
#line 236
void __cdecl perror(const char * _ErrMsg); 
#line 242 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char **__cdecl __p__pgmptr(); 
#line 243
__wchar_t **__cdecl __p__wpgmptr(); 
#line 244
int *__cdecl __p__fmode(); 
#line 259 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
errno_t __cdecl _get_pgmptr(char ** _Value); 
#line 262
errno_t __cdecl _get_wpgmptr(__wchar_t ** _Value); 
#line 264
errno_t __cdecl _set_fmode(int _Mode); 
#line 266
errno_t __cdecl _get_fmode(int * _PMode); 
#line 279
typedef 
#line 275
struct _div_t { 
#line 277
int quot; 
#line 278
int rem; 
#line 279
} div_t; 
#line 285
typedef 
#line 281
struct _ldiv_t { 
#line 283
long quot; 
#line 284
long rem; 
#line 285
} ldiv_t; 
#line 291
typedef 
#line 287
struct _lldiv_t { 
#line 289
__int64 quot; 
#line 290
__int64 rem; 
#line 291
} lldiv_t; 
#line 293
int __cdecl abs(int _Number); 
#line 294
long __cdecl labs(long _Number); 
#line 295
__int64 __cdecl llabs(__int64 _Number); 
#line 296
__int64 __cdecl _abs64(__int64 _Number); 
#line 298
unsigned short __cdecl _byteswap_ushort(unsigned short _Number); 
#line 299
unsigned long __cdecl _byteswap_ulong(unsigned long _Number); 
#line 300
unsigned __int64 __cdecl _byteswap_uint64(unsigned __int64 _Number); 
#line 302
div_t __cdecl div(int _Numerator, int _Denominator); 
#line 303
ldiv_t __cdecl ldiv(long _Numerator, long _Denominator); 
#line 304
lldiv_t __cdecl lldiv(__int64 _Numerator, __int64 _Denominator); 
#line 308
#pragma warning(push)
#pragma warning(disable: 6540)
#line 311
unsigned __cdecl _rotl(unsigned _Value, int _Shift); 
#line 317
unsigned long __cdecl _lrotl(unsigned long _Value, int _Shift); 
#line 322
unsigned __int64 __cdecl _rotl64(unsigned __int64 _Value, int _Shift); 
#line 327
unsigned __cdecl _rotr(unsigned _Value, int _Shift); 
#line 333
unsigned long __cdecl _lrotr(unsigned long _Value, int _Shift); 
#line 338
unsigned __int64 __cdecl _rotr64(unsigned __int64 _Value, int _Shift); 
#line 343
#pragma warning(pop)
#line 350
void __cdecl srand(unsigned _Seed); 
#line 352
int __cdecl rand(); 
#line 361 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
extern "C++" {
#line 363
inline long abs(const long _X) throw() 
#line 364
{ 
#line 365
return labs(_X); 
#line 366
} 
#line 368
inline __int64 abs(const __int64 _X) throw() 
#line 369
{ 
#line 370
return llabs(_X); 
#line 371
} 
#line 373
inline ldiv_t div(const long _A1, const long _A2) throw() 
#line 374
{ 
#line 375
return ldiv(_A1, _A2); 
#line 376
} 
#line 378
inline lldiv_t div(const __int64 _A1, const __int64 _A2) throw() 
#line 379
{ 
#line 380
return lldiv(_A1, _A2); 
#line 381
} 
#line 382
}
#line 394 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
#pragma pack ( push, 4 )
#line 398
typedef 
#line 396
struct { 
#line 397
unsigned char ld[10]; 
#line 398
} _LDOUBLE; 
#pragma pack ( pop )
#line 418 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
typedef 
#line 416
struct { 
#line 417
double x; 
#line 418
} _CRT_DOUBLE; 
#line 423
typedef 
#line 421
struct { 
#line 422
float f; 
#line 423
} _CRT_FLOAT; 
#line 432
typedef 
#line 430
struct { 
#line 431
long double x; 
#line 432
} _LONGDOUBLE; 
#line 436
#pragma pack ( push, 4 )
#line 440
typedef 
#line 438
struct { 
#line 439
unsigned char ld12[12]; 
#line 440
} _LDBL12; 
#pragma pack ( pop )
#line 450
double __cdecl atof(const char * _String); 
#line 451
int __cdecl atoi(const char * _String); 
#line 452
long __cdecl atol(const char * _String); 
#line 453
__int64 __cdecl atoll(const char * _String); 
#line 454
__int64 __cdecl _atoi64(const char * _String); 
#line 456
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 457
int __cdecl _atoi_l(const char * _String, _locale_t _Locale); 
#line 458
long __cdecl _atol_l(const char * _String, _locale_t _Locale); 
#line 459
__int64 __cdecl _atoll_l(const char * _String, _locale_t _Locale); 
#line 460
__int64 __cdecl _atoi64_l(const char * _String, _locale_t _Locale); 
#line 462
int __cdecl _atoflt(_CRT_FLOAT * _Result, const char * _String); 
#line 463
int __cdecl _atodbl(_CRT_DOUBLE * _Result, char * _String); 
#line 464
int __cdecl _atoldbl(_LDOUBLE * _Result, char * _String); 
#line 467
int __cdecl _atoflt_l(_CRT_FLOAT * _Result, const char * _String, _locale_t _Locale); 
#line 474
int __cdecl _atodbl_l(_CRT_DOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 482
int __cdecl _atoldbl_l(_LDOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 489
float __cdecl strtof(const char * _String, char ** _EndPtr); 
#line 495
float __cdecl _strtof_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 502
double __cdecl strtod(const char * _String, char ** _EndPtr); 
#line 508
double __cdecl _strtod_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 515
long double __cdecl strtold(const char * _String, char ** _EndPtr); 
#line 521
long double __cdecl _strtold_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 528
long __cdecl strtol(const char * _String, char ** _EndPtr, int _Radix); 
#line 535
long __cdecl _strtol_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 543
__int64 __cdecl strtoll(const char * _String, char ** _EndPtr, int _Radix); 
#line 550
__int64 __cdecl _strtoll_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 558
unsigned long __cdecl strtoul(const char * _String, char ** _EndPtr, int _Radix); 
#line 565
unsigned long __cdecl _strtoul_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 573
unsigned __int64 __cdecl strtoull(const char * _String, char ** _EndPtr, int _Radix); 
#line 580
unsigned __int64 __cdecl _strtoull_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 588
__int64 __cdecl _strtoi64(const char * _String, char ** _EndPtr, int _Radix); 
#line 595
__int64 __cdecl _strtoi64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 603
unsigned __int64 __cdecl _strtoui64(const char * _String, char ** _EndPtr, int _Radix); 
#line 610
unsigned __int64 __cdecl _strtoui64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 626
errno_t __cdecl _itoa_s(int _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 633
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itoa_s ( int _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 641 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char *__cdecl _itoa(int _Value, char * _Buffer, int _Radix); 
#line 650 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
errno_t __cdecl _ltoa_s(long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 657
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltoa_s ( long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 664 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char *__cdecl _ltoa(long _Value, char * _Buffer, int _Radix); 
#line 673 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
errno_t __cdecl _ultoa_s(unsigned long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 680
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultoa_s ( unsigned long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 687 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char *__cdecl _ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 696 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
errno_t __cdecl _i64toa_s(__int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 705
char *__cdecl _i64toa(__int64 _Value, char * _Buffer, int _Radix); 
#line 713
errno_t __cdecl _ui64toa_s(unsigned __int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 721
char *__cdecl _ui64toa(unsigned __int64 _Value, char * _Buffer, int _Radix); 
#line 741
errno_t __cdecl _ecvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 750
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ecvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _ecvt_s ( _Buffer, _Size, _Value, _DigitCount, _PtDec, _PtSign ); }}
#line 760 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char *__cdecl _ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 769
errno_t __cdecl _fcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 778
extern "C++" {template < size_t _Size > inline errno_t __cdecl _fcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _fcvt_s ( _Buffer, _Size, _Value, _FractionalDigitCount, _PtDec, _PtSign ); }}
#line 790 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char *__cdecl _fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 798
errno_t __cdecl _gcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount); 
#line 805
extern "C++" {template < size_t _Size > inline errno_t __cdecl _gcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount ) throw ( ) { return _gcvt_s ( _Buffer, _Size, _Value, _DigitCount ); }}
#line 814 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char *__cdecl _gcvt(double _Value, int _DigitCount, char * _Buffer); 
#line 852
int __cdecl mblen(const char * _Ch, size_t _MaxCount); 
#line 858
int __cdecl _mblen_l(const char * _Ch, size_t _MaxCount, _locale_t _Locale); 
#line 866
size_t __cdecl _mbstrlen(const char * _String); 
#line 872
size_t __cdecl _mbstrlen_l(const char * _String, _locale_t _Locale); 
#line 879
size_t __cdecl _mbstrnlen(const char * _String, size_t _MaxCount); 
#line 886
size_t __cdecl _mbstrnlen_l(const char * _String, size_t _MaxCount, _locale_t _Locale); 
#line 893
int __cdecl mbtowc(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes); 
#line 900
int __cdecl _mbtowc_l(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes, _locale_t _Locale); 
#line 908
errno_t __cdecl mbstowcs_s(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount); 
#line 916
extern "C++" {template < size_t _Size > inline errno_t __cdecl mbstowcs_s ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount ) throw ( ) { return mbstowcs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 924 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
size_t __cdecl mbstowcs(__wchar_t * _Dest, const char * _Source, size_t _MaxCount); 
#line 932 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
errno_t __cdecl _mbstowcs_s_l(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount, _locale_t _Locale); 
#line 941
extern "C++" {template < size_t _Size > inline errno_t __cdecl _mbstowcs_s_l ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _mbstowcs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 950 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
size_t __cdecl _mbstowcs_l(__wchar_t * _Dest, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 963 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
int __cdecl wctomb(char * _MbCh, __wchar_t _WCh); 
#line 969
int __cdecl _wctomb_l(char * _MbCh, __wchar_t _WCh, _locale_t _Locale); 
#line 978
errno_t __cdecl wctomb_s(int * _SizeConverted, char * _MbCh, rsize_t _SizeInBytes, __wchar_t _WCh); 
#line 988 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
errno_t __cdecl _wctomb_s_l(int * _SizeConverted, char * _MbCh, size_t _SizeInBytes, __wchar_t _WCh, _locale_t _Locale); 
#line 996
errno_t __cdecl wcstombs_s(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes); 
#line 1004
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcstombs_s ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount ) throw ( ) { return wcstombs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 1012 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
size_t __cdecl wcstombs(char * _Dest, const __wchar_t * _Source, size_t _MaxCount); 
#line 1020 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
errno_t __cdecl _wcstombs_s_l(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes, _locale_t _Locale); 
#line 1029
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcstombs_s_l ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _wcstombs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 1038 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
size_t __cdecl _wcstombs_l(char * _Dest, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 1068 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
__declspec(allocator) char *__cdecl _fullpath(char * _Buffer, const char * _Path, size_t _BufferCount); 
#line 1077
errno_t __cdecl _makepath_s(char * _Buffer, size_t _BufferCount, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1086
extern "C++" {template < size_t _Size > inline errno_t __cdecl _makepath_s ( char ( & _Buffer ) [ _Size ], char const * _Drive, char const * _Dir, char const * _Filename, char const * _Ext ) throw ( ) { return _makepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 1095 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
void __cdecl _makepath(char * _Buffer, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1105 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
void __cdecl _splitpath(const char * _FullPath, char * _Drive, char * _Dir, char * _Filename, char * _Ext); 
#line 1114
errno_t __cdecl _splitpath_s(const char * _FullPath, char * _Drive, size_t _DriveCount, char * _Dir, size_t _DirCount, char * _Filename, size_t _FilenameCount, char * _Ext, size_t _ExtCount); 
#line 1126
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _splitpath_s ( char const * _Dest, char ( & _Drive ) [ _DriveSize ], char ( & _Dir ) [ _DirSize ], char ( & _Name ) [ _NameSize ], char ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _splitpath_s ( _Dest, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 1132
errno_t __cdecl getenv_s(size_t * _RequiredCount, char * _Buffer, rsize_t _BufferCount, const char * _VarName); 
#line 1144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
int *__cdecl __p___argc(); 
#line 1145
char ***__cdecl __p___argv(); 
#line 1146
__wchar_t ***__cdecl __p___wargv(); 
#line 1158 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char ***__cdecl __p__environ(); 
#line 1159
__wchar_t ***__cdecl __p__wenviron(); 
#line 1184 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
char *__cdecl getenv(const char * _VarName); 
#line 1188
extern "C++" {template < size_t _Size > inline errno_t __cdecl getenv_s ( size_t * _RequiredCount, char ( & _Buffer ) [ _Size ], char const * _VarName ) throw ( ) { return getenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 1201 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
errno_t __cdecl _dupenv_s(char ** _Buffer, size_t * _BufferCount, const char * _VarName); 
#line 1211 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
int __cdecl system(const char * _Command); 
#line 1217
#pragma warning(push)
#pragma warning(disable: 6540)
#line 1221
int __cdecl _putenv(const char * _EnvString); 
#line 1226
errno_t __cdecl _putenv_s(const char * _Name, const char * _Value); 
#line 1231
#pragma warning(pop)
#line 1233
errno_t __cdecl _searchenv_s(const char * _Filename, const char * _VarName, char * _Buffer, size_t _BufferCount); 
#line 1240
extern "C++" {template < size_t _Size > inline errno_t __cdecl _searchenv_s ( char const * _Filename, char const * _VarName, char ( & _Buffer ) [ _Size ] ) throw ( ) { return _searchenv_s ( _Filename, _VarName, _Buffer, _Size ); }}
#line 1247 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
void __cdecl _searchenv(const char * _Filename, const char * _VarName, char * _Buffer); 
#line 1255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetErrorMode in" "stead. See online help for details.")) void __cdecl 
#line 1256
_seterrormode(int _Mode); 
#line 1260
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Beep instead. S" "ee online help for details.")) void __cdecl 
#line 1261
_beep(unsigned _Frequency, unsigned _Duration); 
#line 1266
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Sleep instead. " "See online help for details.")) void __cdecl 
#line 1267
_sleep(unsigned long _Duration); 
#line 1289 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4141)
#line 1293
char *__cdecl ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 1301
char *__cdecl fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 1309
char *__cdecl gcvt(double _Value, int _DigitCount, char * _DstBuf); 
#line 1316
char *__cdecl itoa(int _Value, char * _Buffer, int _Radix); 
#line 1323
char *__cdecl ltoa(long _Value, char * _Buffer, int _Radix); 
#line 1331
void __cdecl swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 1338
char *__cdecl ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 1347
int __cdecl putenv(const char * _EnvString); 
#line 1351
#pragma warning(pop)
#line 1353
_onexit_t __cdecl onexit(_onexit_t _Func); 
#line 1359 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\stdlib.h"
}__pragma( pack ( pop )) 
#line 1361
#pragma warning(pop)
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
extern "C" {
#line 191
extern cudaError_t __stdcall __cudaDeviceSynchronizeDeprecationAvoidance(); 
#line 244
extern cudaError_t __stdcall __cudaCDP2DeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 245
extern cudaError_t __stdcall __cudaCDP2DeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 246
extern cudaError_t __stdcall __cudaCDP2DeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 247
extern cudaError_t __stdcall __cudaCDP2DeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 248
extern cudaError_t __stdcall __cudaCDP2GetLastError(); 
#line 249
extern cudaError_t __stdcall __cudaCDP2PeekAtLastError(); 
#line 250
extern const char *__stdcall __cudaCDP2GetErrorString(cudaError_t error); 
#line 251
extern const char *__stdcall __cudaCDP2GetErrorName(cudaError_t error); 
#line 252
extern cudaError_t __stdcall __cudaCDP2GetDeviceCount(int * count); 
#line 253
extern cudaError_t __stdcall __cudaCDP2GetDevice(int * device); 
#line 254
extern cudaError_t __stdcall __cudaCDP2StreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 255
extern cudaError_t __stdcall __cudaCDP2StreamDestroy(cudaStream_t stream); 
#line 256
extern cudaError_t __stdcall __cudaCDP2StreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 257
extern cudaError_t __stdcall __cudaCDP2StreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 258
extern cudaError_t __stdcall __cudaCDP2EventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 259
extern cudaError_t __stdcall __cudaCDP2EventRecord(cudaEvent_t event, cudaStream_t stream); 
#line 260
extern cudaError_t __stdcall __cudaCDP2EventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
#line 261
extern cudaError_t __stdcall __cudaCDP2EventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
#line 262
extern cudaError_t __stdcall __cudaCDP2EventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
#line 263
extern cudaError_t __stdcall __cudaCDP2EventDestroy(cudaEvent_t event); 
#line 264
extern cudaError_t __stdcall __cudaCDP2FuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 265
extern cudaError_t __stdcall __cudaCDP2Free(void * devPtr); 
#line 266
extern cudaError_t __stdcall __cudaCDP2Malloc(void ** devPtr, size_t size); 
#line 267
extern cudaError_t __stdcall __cudaCDP2MemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 268
extern cudaError_t __stdcall __cudaCDP2MemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 269
extern cudaError_t __stdcall __cudaCDP2Memcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 270
extern cudaError_t __stdcall __cudaCDP2Memcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 271
extern cudaError_t __stdcall __cudaCDP2Memcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 272
extern cudaError_t __stdcall __cudaCDP2Memcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 273
extern cudaError_t __stdcall __cudaCDP2MemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 274
extern cudaError_t __stdcall __cudaCDP2MemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 275
extern cudaError_t __stdcall __cudaCDP2Memset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 276
extern cudaError_t __stdcall __cudaCDP2Memset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 277
extern cudaError_t __stdcall __cudaCDP2Memset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 278
extern cudaError_t __stdcall __cudaCDP2Memset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 279
extern cudaError_t __stdcall __cudaCDP2RuntimeGetVersion(int * runtimeVersion); 
#line 280
extern void *__stdcall __cudaCDP2GetParameterBuffer(size_t alignment, size_t size); 
#line 281
extern void *__stdcall __cudaCDP2GetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
#line 282
extern cudaError_t __stdcall __cudaCDP2LaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 283
extern cudaError_t __stdcall __cudaCDP2LaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
#line 284
extern cudaError_t __stdcall __cudaCDP2LaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 285
extern cudaError_t __stdcall __cudaCDP2LaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
#line 286
extern cudaError_t __stdcall __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
#line 287
extern cudaError_t __stdcall __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 290
extern cudaError_t __stdcall cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 311 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
static inline cudaGraphExec_t __stdcall cudaGetCurrentGraphExec() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 312
{ 
#line 313
unsigned __int64 current_graph_exec; 
#line 314
__asm mov.u64 %0, %%current_graph_exec;
return (cudaGraphExec_t)current_graph_exec; 
#line 316
} 
#endif
#line 346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t node, size_t offset, const void * value, size_t size); 
#line 374
extern cudaError_t __stdcall cudaGraphKernelNodeSetEnabled(cudaGraphDeviceNode_t node, bool enable); 
#line 401
extern cudaError_t __stdcall cudaGraphKernelNodeSetGridDim(cudaGraphDeviceNode_t node, dim3 gridDim); 
#line 430
extern cudaError_t __stdcall cudaGraphKernelNodeUpdatesApply(const cudaGraphKernelNodeUpdate * updates, size_t updateCount); 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 448
static inline void __stdcall cudaTriggerProgrammaticLaunchCompletion() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 449
{ 
#line 450
__asm griddepcontrol.launch_dependents;
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 464 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
static inline void __stdcall cudaGridDependencySynchronize() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 465
{ 
#line 466
__asm griddepcontrol.wait;
} 
#endif
#line 479 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
extern void __stdcall cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned value); 
#line 482
extern unsigned __int64 __stdcall cudaCGGetIntrinsicHandle(cudaCGScope scope); 
#line 483
extern cudaError_t __stdcall cudaCGSynchronize(unsigned __int64 handle, unsigned flags); 
#line 484
extern cudaError_t __stdcall cudaCGSynchronizeGrid(unsigned __int64 handle, unsigned flags); 
#line 485
extern cudaError_t __stdcall cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned __int64 handle); 
#line 486
extern cudaError_t __stdcall cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned __int64 handle); 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 714
static __inline void *__stdcall cudaGetParameterBuffer(size_t alignment, size_t size) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)alignment;(void)size;::exit(___);}
#if 0
#line 715
{ 
#line 716
return __cudaCDP2GetParameterBuffer(alignment, size); 
#line 717
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 724 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
static __inline void *__stdcall cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)func;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;::exit(___);}
#if 0
#line 725
{ 
#line 726
return __cudaCDP2GetParameterBufferV2(func, gridDimension, blockDimension, sharedMemSize); 
#line 727
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 734 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
static __inline cudaError_t __stdcall cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)func;(void)parameterBuffer;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;(void)stream;::exit(___);}
#if 0
#line 735
{ 
#line 736
return __cudaCDP2LaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream); 
#line 737
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 739 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
static __inline cudaError_t __stdcall cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)parameterBuffer;(void)stream;::exit(___);}
#if 0
#line 740
{ 
#line 741
return __cudaCDP2LaunchDeviceV2_ptsz(parameterBuffer, stream); 
#line 742
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 800 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
static __inline cudaError_t __stdcall cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)func;(void)parameterBuffer;(void)gridDimension;(void)blockDimension;(void)sharedMemSize;(void)stream;::exit(___);}
#if 0
#line 801
{ 
#line 802
return __cudaCDP2LaunchDevice(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream); 
#line 803
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 805 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
static __inline cudaError_t __stdcall cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)parameterBuffer;(void)stream;::exit(___);}
#if 0
#line 806
{ 
#line 807
return __cudaCDP2LaunchDeviceV2(parameterBuffer, stream); 
#line 808
} 
#endif
#line 862 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_device_runtime_api.h"
}
#line 868
template< class T> static __inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
#line 869
template< class T> static __inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
#line 870
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
#line 871
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 901
template< class T> static __inline ::cudaError_t __stdcall 
#line 902
cudaGraphKernelNodeSetParam(::cudaGraphDeviceNode_t node, ::size_t offset, const T &value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)node;(void)offset;(void)value;::exit(___);}
#if 0
#line 903
{ 
#line 904
return cudaGraphKernelNodeSetParam(node, offset, &value, sizeof(T)); 
#line 905
} 
#endif
#line 287 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern "C" {
#line 334 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceReset(); 
#line 356
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 442
extern cudaError_t __stdcall cudaDeviceSetLimit(cudaLimit limit, size_t value); 
#line 478
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 501
extern cudaError_t __stdcall cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
#line 535 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 572
extern cudaError_t __stdcall cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
#line 616
extern cudaError_t __stdcall cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
#line 643
extern cudaError_t __stdcall cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
#line 673
extern cudaError_t __stdcall cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
#line 724
extern cudaError_t __stdcall cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
#line 768
extern cudaError_t __stdcall cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
#line 813
extern cudaError_t __stdcall cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
#line 880
extern cudaError_t __stdcall cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
#line 919
extern cudaError_t __stdcall cudaIpcCloseMemHandle(void * devPtr); 
#line 951
extern cudaError_t __stdcall cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope); 
#line 989 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void * userData, cudaAsyncCallbackHandle_t * callback); 
#line 1012
extern cudaError_t __stdcall cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback); 
#line 1059
__declspec(deprecated) extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 1105
__declspec(deprecated) extern cudaError_t __stdcall cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
#line 1146
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadExit(); 
#line 1172
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSynchronize(); 
#line 1221
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSetLimit(cudaLimit limit, size_t value); 
#line 1254
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
#line 1290
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 1337
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
#line 1402
extern cudaError_t __stdcall cudaGetLastError(); 
#line 1453
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 1469
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 1485
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 1514
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 1818
extern cudaError_t __stdcall cudaGetDeviceProperties_v2(cudaDeviceProp * prop, int device); 
#line 2024
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 2042
extern cudaError_t __stdcall cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int device); 
#line 2066
extern cudaError_t __stdcall cudaDeviceSetMemPool(int device, cudaMemPool_t memPool); 
#line 2086
extern cudaError_t __stdcall cudaDeviceGetMemPool(cudaMemPool_t * memPool, int device); 
#line 2148
extern cudaError_t __stdcall cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
#line 2188
extern cudaError_t __stdcall cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
#line 2210
extern cudaError_t __stdcall cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
#line 2239
extern cudaError_t __stdcall cudaInitDevice(int device, unsigned deviceFlags, unsigned flags); 
#line 2285
extern cudaError_t __stdcall cudaSetDevice(int device); 
#line 2307
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 2338
extern cudaError_t __stdcall cudaSetValidDevices(int * device_arr, int len); 
#line 2408
extern cudaError_t __stdcall cudaSetDeviceFlags(unsigned flags); 
#line 2453
extern cudaError_t __stdcall cudaGetDeviceFlags(unsigned * flags); 
#line 2497
extern cudaError_t __stdcall cudaStreamCreate(cudaStream_t * pStream); 
#line 2533
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 2585
extern cudaError_t __stdcall cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
#line 2613
extern cudaError_t __stdcall cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
#line 2639
extern cudaError_t __stdcall cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
#line 2676
extern cudaError_t __stdcall cudaStreamGetId(cudaStream_t hStream, unsigned __int64 * streamId); 
#line 2702
extern cudaError_t __stdcall cudaStreamGetDevice(cudaStream_t hStream, int * device); 
#line 2717
extern cudaError_t __stdcall cudaCtxResetPersistingL2Cache(); 
#line 2737
extern cudaError_t __stdcall cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
#line 2758
extern cudaError_t __stdcall cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
#line 2782
extern cudaError_t __stdcall cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
#line 2816
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 2847
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
#line 2855
typedef void (__stdcall *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
#line 2922
extern cudaError_t __stdcall cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
#line 2946
extern cudaError_t __stdcall cudaStreamSynchronize(cudaStream_t stream); 
#line 2971
extern cudaError_t __stdcall cudaStreamQuery(cudaStream_t stream); 
#line 3055
extern cudaError_t __stdcall cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
#line 3094 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
#line 3135
extern cudaError_t __stdcall cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t * dependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, cudaStreamCaptureMode mode); 
#line 3186
extern cudaError_t __stdcall cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
#line 3215
extern cudaError_t __stdcall cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
#line 3253
extern cudaError_t __stdcall cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
#line 3303
extern cudaError_t __stdcall cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned __int64 * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, size_t * numDependencies_out = 0); 
#line 3362
extern cudaError_t __stdcall cudaStreamGetCaptureInfo_v3(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned __int64 * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, const cudaGraphEdgeData ** edgeData_out = 0, size_t * numDependencies_out = 0); 
#line 3402
extern cudaError_t __stdcall cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t * dependencies, size_t numDependencies, unsigned flags = 0); 
#line 3437
extern cudaError_t __stdcall cudaStreamUpdateCaptureDependencies_v2(cudaStream_t stream, cudaGraphNode_t * dependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, unsigned flags = 0); 
#line 3474
extern cudaError_t __stdcall cudaEventCreate(cudaEvent_t * event); 
#line 3511
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 3552
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
#line 3600
extern cudaError_t __stdcall cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
#line 3633 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaEventQuery(cudaEvent_t event); 
#line 3664
extern cudaError_t __stdcall cudaEventSynchronize(cudaEvent_t event); 
#line 3694
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 3739
extern cudaError_t __stdcall cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
#line 3787
extern cudaError_t __stdcall cudaEventElapsedTime_v2(float * ms, cudaEvent_t start, cudaEvent_t end); 
#line 3968
extern cudaError_t __stdcall cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
#line 4023
extern cudaError_t __stdcall cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
#line 4083
extern cudaError_t __stdcall cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
#line 4107
extern cudaError_t __stdcall cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
#line 4261
extern cudaError_t __stdcall cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
#line 4344
extern cudaError_t __stdcall cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
#line 4420
extern cudaError_t __stdcall cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
#line 4443
extern cudaError_t __stdcall cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
#line 4511
extern cudaError_t __stdcall cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 4574
extern cudaError_t __stdcall cudaLaunchKernelExC(const cudaLaunchConfig_t * config, const void * func, void ** args); 
#line 4632
extern cudaError_t __stdcall cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 4733
__declspec(deprecated) extern cudaError_t __stdcall cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
#line 4783
extern cudaError_t __stdcall cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
#line 4818
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 4877
extern cudaError_t __stdcall cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
#line 4903
extern cudaError_t __stdcall cudaFuncGetName(const char ** name, const void * func); 
#line 4926
extern cudaError_t __stdcall cudaFuncGetParamInfo(const void * func, size_t paramIndex, size_t * paramOffset, size_t * paramSize); 
#line 4950
__declspec(deprecated) extern cudaError_t __stdcall cudaSetDoubleForDevice(double * d); 
#line 4974
__declspec(deprecated) extern cudaError_t __stdcall cudaSetDoubleForHost(double * d); 
#line 5040
extern cudaError_t __stdcall cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
#line 5114
__declspec(deprecated) extern cudaError_t __stdcall cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
#line 5171
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
#line 5201
extern cudaError_t __stdcall cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
#line 5247
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
#line 5283
extern cudaError_t __stdcall cudaOccupancyMaxPotentialClusterSize(int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig); 
#line 5323
extern cudaError_t __stdcall cudaOccupancyMaxActiveClusters(int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig); 
#line 5443
extern cudaError_t __stdcall cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
#line 5476 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 5513
extern cudaError_t __stdcall cudaMallocHost(void ** ptr, size_t size); 
#line 5556
extern cudaError_t __stdcall cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
#line 5608
extern cudaError_t __stdcall cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
#line 5647
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 5670
extern cudaError_t __stdcall cudaFreeHost(void * ptr); 
#line 5693
extern cudaError_t __stdcall cudaFreeArray(cudaArray_t array); 
#line 5716
extern cudaError_t __stdcall cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
#line 5782
extern cudaError_t __stdcall cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
#line 5879
extern cudaError_t __stdcall cudaHostRegister(void * ptr, size_t size, unsigned flags); 
#line 5902
extern cudaError_t __stdcall cudaHostUnregister(void * ptr); 
#line 5947
extern cudaError_t __stdcall cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
#line 5969
extern cudaError_t __stdcall cudaHostGetFlags(unsigned * pFlags, void * pHost); 
#line 6008
extern cudaError_t __stdcall cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
#line 6153
extern cudaError_t __stdcall cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
#line 6298
extern cudaError_t __stdcall cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
#line 6331
extern cudaError_t __stdcall cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
#line 6436
extern cudaError_t __stdcall cudaMemcpy3D(const cudaMemcpy3DParms * p); 
#line 6468
extern cudaError_t __stdcall cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
#line 6586
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
#line 6613
extern cudaError_t __stdcall cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
#line 6647
extern cudaError_t __stdcall cudaMemGetInfo(size_t * free, size_t * total); 
#line 6673
extern cudaError_t __stdcall cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
#line 6702
extern cudaError_t __stdcall cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t hArray, unsigned planeIdx); 
#line 6725
extern cudaError_t __stdcall cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t array, int device); 
#line 6749
extern cudaError_t __stdcall cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t mipmap, int device); 
#line 6777
extern cudaError_t __stdcall cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
#line 6807 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
#line 6852 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 6887
extern cudaError_t __stdcall cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
#line 6936
extern cudaError_t __stdcall cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 6986
extern cudaError_t __stdcall cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 7036
extern cudaError_t __stdcall cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
#line 7083
extern cudaError_t __stdcall cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 7126
extern cudaError_t __stdcall cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
#line 7170
extern cudaError_t __stdcall cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
#line 7227
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7262
extern cudaError_t __stdcall cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
#line 7334
extern cudaError_t __stdcall cudaMemcpyBatchAsync(void ** dsts, void ** srcs, size_t * sizes, size_t count, cudaMemcpyAttributes * attrs, size_t * attrsIdxs, size_t numAttrs, size_t * failIdx, cudaStream_t stream); 
#line 7402
extern cudaError_t __stdcall cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp * opList, size_t * failIdx, unsigned __int64 flags, cudaStream_t stream); 
#line 7465
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7523
extern cudaError_t __stdcall cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7580
extern cudaError_t __stdcall cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7631
extern cudaError_t __stdcall cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7682
extern cudaError_t __stdcall cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7711
extern cudaError_t __stdcall cudaMemset(void * devPtr, int value, size_t count); 
#line 7745
extern cudaError_t __stdcall cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
#line 7791
extern cudaError_t __stdcall cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
#line 7827
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
#line 7868
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
#line 7921
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
#line 7949
extern cudaError_t __stdcall cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
#line 7976
extern cudaError_t __stdcall cudaGetSymbolSize(size_t * size, const void * symbol); 
#line 8048
extern cudaError_t __stdcall cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
#line 8129
extern cudaError_t __stdcall cudaMemPrefetchAsync_v2(const void * devPtr, size_t count, cudaMemLocation location, unsigned flags, cudaStream_t stream = 0); 
#line 8243
extern cudaError_t __stdcall cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
#line 8366
extern cudaError_t __stdcall cudaMemAdvise_v2(const void * devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location); 
#line 8448
extern cudaError_t __stdcall cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
#line 8491
extern cudaError_t __stdcall cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
#line 8551
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
#line 8593
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
#line 8636
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 8687
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 8737
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 8806
extern cudaError_t __stdcall cudaMallocAsync(void ** devPtr, size_t size, cudaStream_t hStream); 
#line 8832
extern cudaError_t __stdcall cudaFreeAsync(void * devPtr, cudaStream_t hStream); 
#line 8857
extern cudaError_t __stdcall cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep); 
#line 8901
extern cudaError_t __stdcall cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
#line 8949
extern cudaError_t __stdcall cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
#line 8964
extern cudaError_t __stdcall cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc * descList, size_t count); 
#line 8977
extern cudaError_t __stdcall cudaMemPoolGetAccess(cudaMemAccessFlags * flags, cudaMemPool_t memPool, cudaMemLocation * location); 
#line 9021
extern cudaError_t __stdcall cudaMemPoolCreate(cudaMemPool_t * memPool, const cudaMemPoolProps * poolProps); 
#line 9043
extern cudaError_t __stdcall cudaMemPoolDestroy(cudaMemPool_t memPool); 
#line 9079
extern cudaError_t __stdcall cudaMallocFromPoolAsync(void ** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream); 
#line 9104
extern cudaError_t __stdcall cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned flags); 
#line 9131
extern cudaError_t __stdcall cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, cudaMemAllocationHandleType handleType, unsigned flags); 
#line 9154
extern cudaError_t __stdcall cudaMemPoolExportPointer(cudaMemPoolPtrExportData * exportData, void * ptr); 
#line 9183
extern cudaError_t __stdcall cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData * exportData); 
#line 9336
extern cudaError_t __stdcall cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
#line 9377
extern cudaError_t __stdcall cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
#line 9419
extern cudaError_t __stdcall cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
#line 9441
extern cudaError_t __stdcall cudaDeviceDisablePeerAccess(int peerDevice); 
#line 9505
extern cudaError_t __stdcall cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
#line 9540
extern cudaError_t __stdcall cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
#line 9579
extern cudaError_t __stdcall cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 9614
extern cudaError_t __stdcall cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 9646
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
#line 9684
extern cudaError_t __stdcall cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
#line 9713
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
#line 9748
extern cudaError_t __stdcall cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
#line 9778
extern cudaChannelFormatDesc __stdcall cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
#line 10003
extern cudaError_t __stdcall cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
#line 10023
extern cudaError_t __stdcall cudaDestroyTextureObject(cudaTextureObject_t texObject); 
#line 10043
extern cudaError_t __stdcall cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
#line 10063
extern cudaError_t __stdcall cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
#line 10084
extern cudaError_t __stdcall cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
#line 10129
extern cudaError_t __stdcall cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
#line 10149
extern cudaError_t __stdcall cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
#line 10168
extern cudaError_t __stdcall cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
#line 10202
extern cudaError_t __stdcall cudaDriverGetVersion(int * driverVersion); 
#line 10231
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 10278
extern cudaError_t __stdcall cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
#line 10377
extern cudaError_t __stdcall cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
#line 10410
extern cudaError_t __stdcall cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
#line 10437
extern cudaError_t __stdcall cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
#line 10457
extern cudaError_t __stdcall cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst); 
#line 10480
extern cudaError_t __stdcall cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
#line 10504
extern cudaError_t __stdcall cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
#line 10555
extern cudaError_t __stdcall cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
#line 10614
extern cudaError_t __stdcall cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10683 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10751 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 10783 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
#line 10810
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
#line 10849
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10895 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10941 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 10989 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
#line 11012
extern cudaError_t __stdcall cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
#line 11036
extern cudaError_t __stdcall cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
#line 11078
extern cudaError_t __stdcall cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
#line 11101
extern cudaError_t __stdcall cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
#line 11125
extern cudaError_t __stdcall cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
#line 11166
extern cudaError_t __stdcall cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
#line 11193
extern cudaError_t __stdcall cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
#line 11231
extern cudaError_t __stdcall cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
#line 11275
extern cudaError_t __stdcall cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
#line 11302 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
#line 11330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
#line 11377 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
#line 11404 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
#line 11432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
#line 11482 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 11515 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out); 
#line 11543 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 11593 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 11626 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out); 
#line 11654 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 11732 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams); 
#line 11759 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out); 
#line 11820 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dptr); 
#line 11844 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void * dptr_out); 
#line 11872 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGraphMemTrim(int device); 
#line 11909 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
#line 11943 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
#line 11971 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
#line 11999
extern cudaError_t __stdcall cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
#line 12030
extern cudaError_t __stdcall cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
#line 12061
extern cudaError_t __stdcall cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
#line 12092
extern cudaError_t __stdcall cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
#line 12126
extern cudaError_t __stdcall cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
#line 12166
extern cudaError_t __stdcall cudaGraphGetEdges_v2(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, cudaGraphEdgeData * edgeData, size_t * numEdges); 
#line 12197
extern cudaError_t __stdcall cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
#line 12234
extern cudaError_t __stdcall cudaGraphNodeGetDependencies_v2(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, cudaGraphEdgeData * edgeData, size_t * pNumDependencies); 
#line 12266
extern cudaError_t __stdcall cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
#line 12304
extern cudaError_t __stdcall cudaGraphNodeGetDependentNodes_v2(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, cudaGraphEdgeData * edgeData, size_t * pNumDependentNodes); 
#line 12335
extern cudaError_t __stdcall cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
#line 12367
extern cudaError_t __stdcall cudaGraphAddDependencies_v2(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, const cudaGraphEdgeData * edgeData, size_t numDependencies); 
#line 12398
extern cudaError_t __stdcall cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
#line 12433
extern cudaError_t __stdcall cudaGraphRemoveDependencies_v2(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, const cudaGraphEdgeData * edgeData, size_t numDependencies); 
#line 12463
extern cudaError_t __stdcall cudaGraphDestroyNode(cudaGraphNode_t node); 
#line 12534
extern cudaError_t __stdcall cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned __int64 flags = 0); 
#line 12607
extern cudaError_t __stdcall cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned __int64 flags = 0); 
#line 12714 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphInstantiateWithParams(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams * instantiateParams); 
#line 12739
extern cudaError_t __stdcall cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned __int64 * flags); 
#line 12799
extern cudaError_t __stdcall cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
#line 12850
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
#line 12905
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 12968 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 13029 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 13088 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
#line 13128
extern cudaError_t __stdcall cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
#line 13175
extern cudaError_t __stdcall cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
#line 13220 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
#line 13265 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
#line 13313 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 13361 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 13401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned isEnabled); 
#line 13435 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned * isEnabled); 
#line 13529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo * resultInfo); 
#line 13554
extern cudaError_t __stdcall cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 13585 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 13608
extern cudaError_t __stdcall cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
#line 13629
extern cudaError_t __stdcall cudaGraphDestroy(cudaGraph_t graph); 
#line 13648
extern cudaError_t __stdcall cudaGraphDebugDotPrint(cudaGraph_t graph, const char * path, unsigned flags); 
#line 13684
extern cudaError_t __stdcall cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t destroy, unsigned initialRefcount, unsigned flags); 
#line 13708
extern cudaError_t __stdcall cudaUserObjectRetain(cudaUserObject_t object, unsigned count = 1); 
#line 13736
extern cudaError_t __stdcall cudaUserObjectRelease(cudaUserObject_t object, unsigned count = 1); 
#line 13764
extern cudaError_t __stdcall cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1, unsigned flags = 0); 
#line 13789
extern cudaError_t __stdcall cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1); 
#line 13831
extern cudaError_t __stdcall cudaGraphAddNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraphNodeParams * nodeParams); 
#line 13875
extern cudaError_t __stdcall cudaGraphAddNode_v2(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, const cudaGraphEdgeData * dependencyData, size_t numDependencies, cudaGraphNodeParams * nodeParams); 
#line 13904
extern cudaError_t __stdcall cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams * nodeParams); 
#line 13953
extern cudaError_t __stdcall cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams * nodeParams); 
#line 13980
extern cudaError_t __stdcall cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle * pHandle_out, cudaGraph_t graph, unsigned defaultLaunchValue = 0, unsigned flags = 0); 
#line 14061
extern cudaError_t __stdcall cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned __int64 flags, cudaDriverEntryPointQueryResult * driverStatus = 0); 
#line 14137 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGetDriverEntryPointByVersion(const char * symbol, void ** funcPtr, unsigned cudaVersion, unsigned __int64 flags, cudaDriverEntryPointQueryResult * driverStatus = 0); 
#line 14212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaLibraryLoadData(cudaLibrary_t * library, const void * code, cudaJitOption * jitOptions, void ** jitOptionsValues, unsigned numJitOptions, cudaLibraryOption * libraryOptions, void ** libraryOptionValues, unsigned numLibraryOptions); 
#line 14272
extern cudaError_t __stdcall cudaLibraryLoadFromFile(cudaLibrary_t * library, const char * fileName, cudaJitOption * jitOptions, void ** jitOptionsValues, unsigned numJitOptions, cudaLibraryOption * libraryOptions, void ** libraryOptionValues, unsigned numLibraryOptions); 
#line 14293
extern cudaError_t __stdcall cudaLibraryUnload(cudaLibrary_t library); 
#line 14318
extern cudaError_t __stdcall cudaLibraryGetKernel(cudaKernel_t * pKernel, cudaLibrary_t library, const char * name); 
#line 14352
extern cudaError_t __stdcall cudaLibraryGetGlobal(void ** dptr, size_t * bytes, cudaLibrary_t library, const char * name); 
#line 14385
extern cudaError_t __stdcall cudaLibraryGetManaged(void ** dptr, size_t * bytes, cudaLibrary_t library, const char * name); 
#line 14412
extern cudaError_t __stdcall cudaLibraryGetUnifiedFunction(void ** fptr, cudaLibrary_t library, const char * symbol); 
#line 14434
extern cudaError_t __stdcall cudaLibraryGetKernelCount(unsigned * count, cudaLibrary_t lib); 
#line 14456
extern cudaError_t __stdcall cudaLibraryEnumerateKernels(cudaKernel_t * kernels, unsigned numKernels, cudaLibrary_t lib); 
#line 14525
extern cudaError_t __stdcall cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device); 
#line 14530
extern cudaError_t __stdcall cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
#line 14716
extern cudaError_t cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
#line 14740
extern cudaError_t __stdcall cudaGetKernel(cudaKernel_t * kernelPtr, const void * entryFuncAddr); 
#line 14918 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\cuda_runtime_api.h"
}
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\channel_descriptor.h"
template< class T> __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 121
{ 
#line 122
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 123
} 
#line 125
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
#line 126
{ 
#line 127
int e = (((int)sizeof(unsigned short)) * 8); 
#line 129
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 130
} 
#line 132
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
#line 133
{ 
#line 134
int e = (((int)sizeof(unsigned short)) * 8); 
#line 136
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 137
} 
#line 139
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
#line 140
{ 
#line 141
int e = (((int)sizeof(unsigned short)) * 8); 
#line 143
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 144
} 
#line 146
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
#line 147
{ 
#line 148
int e = (((int)sizeof(unsigned short)) * 8); 
#line 150
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 151
} 
#line 153
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
#line 154
{ 
#line 155
int e = (((int)sizeof(char)) * 8); 
#line 160 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\channel_descriptor.h"
} 
#line 164
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
#line 165
{ 
#line 166
int e = (((int)sizeof(signed char)) * 8); 
#line 168
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 169
} 
#line 171
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
#line 172
{ 
#line 173
int e = (((int)sizeof(unsigned char)) * 8); 
#line 175
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 176
} 
#line 178
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
#line 179
{ 
#line 180
int e = (((int)sizeof(signed char)) * 8); 
#line 182
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 183
} 
#line 185
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
#line 186
{ 
#line 187
int e = (((int)sizeof(unsigned char)) * 8); 
#line 189
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 190
} 
#line 192
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
#line 193
{ 
#line 194
int e = (((int)sizeof(signed char)) * 8); 
#line 196
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 197
} 
#line 199
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
#line 200
{ 
#line 201
int e = (((int)sizeof(unsigned char)) * 8); 
#line 203
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 204
} 
#line 206
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
#line 207
{ 
#line 208
int e = (((int)sizeof(signed char)) * 8); 
#line 210
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 211
} 
#line 213
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
#line 214
{ 
#line 215
int e = (((int)sizeof(unsigned char)) * 8); 
#line 217
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 218
} 
#line 220
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
#line 221
{ 
#line 222
int e = (((int)sizeof(short)) * 8); 
#line 224
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 225
} 
#line 227
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
#line 228
{ 
#line 229
int e = (((int)sizeof(unsigned short)) * 8); 
#line 231
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 232
} 
#line 234
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
#line 235
{ 
#line 236
int e = (((int)sizeof(short)) * 8); 
#line 238
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 239
} 
#line 241
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
#line 242
{ 
#line 243
int e = (((int)sizeof(unsigned short)) * 8); 
#line 245
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 246
} 
#line 248
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
#line 249
{ 
#line 250
int e = (((int)sizeof(short)) * 8); 
#line 252
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 253
} 
#line 255
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
#line 256
{ 
#line 257
int e = (((int)sizeof(unsigned short)) * 8); 
#line 259
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 260
} 
#line 262
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
#line 263
{ 
#line 264
int e = (((int)sizeof(short)) * 8); 
#line 266
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 267
} 
#line 269
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
#line 270
{ 
#line 271
int e = (((int)sizeof(unsigned short)) * 8); 
#line 273
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 274
} 
#line 276
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
#line 277
{ 
#line 278
int e = (((int)sizeof(int)) * 8); 
#line 280
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 281
} 
#line 283
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
#line 284
{ 
#line 285
int e = (((int)sizeof(unsigned)) * 8); 
#line 287
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 288
} 
#line 290
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
#line 291
{ 
#line 292
int e = (((int)sizeof(int)) * 8); 
#line 294
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 295
} 
#line 297
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
#line 298
{ 
#line 299
int e = (((int)sizeof(unsigned)) * 8); 
#line 301
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 302
} 
#line 304
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
#line 305
{ 
#line 306
int e = (((int)sizeof(int)) * 8); 
#line 308
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 309
} 
#line 311
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
#line 312
{ 
#line 313
int e = (((int)sizeof(unsigned)) * 8); 
#line 315
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 316
} 
#line 318
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
#line 319
{ 
#line 320
int e = (((int)sizeof(int)) * 8); 
#line 322
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 323
} 
#line 325
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
#line 326
{ 
#line 327
int e = (((int)sizeof(unsigned)) * 8); 
#line 329
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 330
} 
#line 334
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long> () 
#line 335
{ 
#line 336
int e = (((int)sizeof(long)) * 8); 
#line 338
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 339
} 
#line 341
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned long> () 
#line 342
{ 
#line 343
int e = (((int)sizeof(unsigned long)) * 8); 
#line 345
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 346
} 
#line 348
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long1> () 
#line 349
{ 
#line 350
int e = (((int)sizeof(long)) * 8); 
#line 352
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 353
} 
#line 355
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong1> () 
#line 356
{ 
#line 357
int e = (((int)sizeof(unsigned long)) * 8); 
#line 359
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 360
} 
#line 362
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long2> () 
#line 363
{ 
#line 364
int e = (((int)sizeof(long)) * 8); 
#line 366
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 367
} 
#line 369
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong2> () 
#line 370
{ 
#line 371
int e = (((int)sizeof(unsigned long)) * 8); 
#line 373
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 374
} 
#line 376
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long4> () 
#line 377
{ 
#line 378
int e = (((int)sizeof(long)) * 8); 
#line 380
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 381
} 
#line 383
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong4> () 
#line 384
{ 
#line 385
int e = (((int)sizeof(unsigned long)) * 8); 
#line 387
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 388
} 
#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\channel_descriptor.h"
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
#line 393
{ 
#line 394
int e = (((int)sizeof(float)) * 8); 
#line 396
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 397
} 
#line 399
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
#line 400
{ 
#line 401
int e = (((int)sizeof(float)) * 8); 
#line 403
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 404
} 
#line 406
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
#line 407
{ 
#line 408
int e = (((int)sizeof(float)) * 8); 
#line 410
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 411
} 
#line 413
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
#line 414
{ 
#line 415
int e = (((int)sizeof(float)) * 8); 
#line 417
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 418
} 
#line 420
static __inline cudaChannelFormatDesc cudaCreateChannelDescNV12() 
#line 421
{ 
#line 422
int e = (((int)sizeof(char)) * 8); 
#line 424
return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12); 
#line 425
} 
#line 427
template< cudaChannelFormatKind > __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 428
{ 
#line 429
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 430
} 
#line 433
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X1> () 
#line 434
{ 
#line 435
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedNormalized8X1); 
#line 436
} 
#line 438
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X2> () 
#line 439
{ 
#line 440
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedNormalized8X2); 
#line 441
} 
#line 443
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X4> () 
#line 444
{ 
#line 445
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSignedNormalized8X4); 
#line 446
} 
#line 449
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X1> () 
#line 450
{ 
#line 451
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized8X1); 
#line 452
} 
#line 454
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X2> () 
#line 455
{ 
#line 456
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedNormalized8X2); 
#line 457
} 
#line 459
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X4> () 
#line 460
{ 
#line 461
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4); 
#line 462
} 
#line 465
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X1> () 
#line 466
{ 
#line 467
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSignedNormalized16X1); 
#line 468
} 
#line 470
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X2> () 
#line 471
{ 
#line 472
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSignedNormalized16X2); 
#line 473
} 
#line 475
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X4> () 
#line 476
{ 
#line 477
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSignedNormalized16X4); 
#line 478
} 
#line 481
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X1> () 
#line 482
{ 
#line 483
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized16X1); 
#line 484
} 
#line 486
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X2> () 
#line 487
{ 
#line 488
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsignedNormalized16X2); 
#line 489
} 
#line 491
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X4> () 
#line 492
{ 
#line 493
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsignedNormalized16X4); 
#line 494
} 
#line 497
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindNV12> () 
#line 498
{ 
#line 499
return cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindNV12); 
#line 500
} 
#line 503
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized1010102> () 
#line 504
{ 
#line 505
return cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102); 
#line 506
} 
#line 509
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1> () 
#line 510
{ 
#line 511
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1); 
#line 512
} 
#line 515
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1SRGB> () 
#line 516
{ 
#line 517
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1SRGB); 
#line 518
} 
#line 521
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2> () 
#line 522
{ 
#line 523
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2); 
#line 524
} 
#line 527
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2SRGB> () 
#line 528
{ 
#line 529
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2SRGB); 
#line 530
} 
#line 533
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3> () 
#line 534
{ 
#line 535
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3); 
#line 536
} 
#line 539
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3SRGB> () 
#line 540
{ 
#line 541
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3SRGB); 
#line 542
} 
#line 545
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed4> () 
#line 546
{ 
#line 547
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed4); 
#line 548
} 
#line 551
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed4> () 
#line 552
{ 
#line 553
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedBlockCompressed4); 
#line 554
} 
#line 557
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed5> () 
#line 558
{ 
#line 559
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed5); 
#line 560
} 
#line 563
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed5> () 
#line 564
{ 
#line 565
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedBlockCompressed5); 
#line 566
} 
#line 569
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed6H> () 
#line 570
{ 
#line 571
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindUnsignedBlockCompressed6H); 
#line 572
} 
#line 575
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed6H> () 
#line 576
{ 
#line 577
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindSignedBlockCompressed6H); 
#line 578
} 
#line 581
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7> () 
#line 582
{ 
#line 583
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7); 
#line 584
} 
#line 587
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7SRGB> () 
#line 588
{ 
#line 589
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7SRGB); 
#line 590
} 
#line 79 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\driver_functions.h"
static __inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
#line 80
{ 
#line 81
cudaPitchedPtr s; 
#line 83
(s.ptr) = d; 
#line 84
(s.pitch) = p; 
#line 85
(s.xsize) = xsz; 
#line 86
(s.ysize) = ysz; 
#line 88
return s; 
#line 89
} 
#line 106
static __inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
#line 107
{ 
#line 108
cudaPos p; 
#line 110
(p.x) = x; 
#line 111
(p.y) = y; 
#line 112
(p.z) = z; 
#line 114
return p; 
#line 115
} 
#line 132
static __inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
#line 133
{ 
#line 134
cudaExtent e; 
#line 136
(e.width) = w; 
#line 137
(e.height) = h; 
#line 138
(e.depth) = d; 
#line 140
return e; 
#line 141
} 
#line 77 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_functions.h"
static __inline char1 make_char1(signed char x); 
#line 79
static __inline uchar1 make_uchar1(unsigned char x); 
#line 81
static __inline char2 make_char2(signed char x, signed char y); 
#line 83
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
#line 85
static __inline char3 make_char3(signed char x, signed char y, signed char z); 
#line 87
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
#line 89
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
#line 91
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
#line 93
static __inline short1 make_short1(short x); 
#line 95
static __inline ushort1 make_ushort1(unsigned short x); 
#line 97
static __inline short2 make_short2(short x, short y); 
#line 99
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
#line 101
static __inline short3 make_short3(short x, short y, short z); 
#line 103
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
#line 105
static __inline short4 make_short4(short x, short y, short z, short w); 
#line 107
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
#line 109
static __inline int1 make_int1(int x); 
#line 111
static __inline uint1 make_uint1(unsigned x); 
#line 113
static __inline int2 make_int2(int x, int y); 
#line 115
static __inline uint2 make_uint2(unsigned x, unsigned y); 
#line 117
static __inline int3 make_int3(int x, int y, int z); 
#line 119
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
#line 121
static __inline int4 make_int4(int x, int y, int z, int w); 
#line 123
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
#line 125
static __inline long1 make_long1(long x); 
#line 127
static __inline ulong1 make_ulong1(unsigned long x); 
#line 129
static __inline long2 make_long2(long x, long y); 
#line 131
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
#line 133
static __inline long3 make_long3(long x, long y, long z); 
#line 135
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
#line 137
static __inline long4 make_long4(long x, long y, long z, long w); 
#line 139
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
#line 141
static __inline float1 make_float1(float x); 
#line 143
static __inline float2 make_float2(float x, float y); 
#line 145
static __inline float3 make_float3(float x, float y, float z); 
#line 147
static __inline float4 make_float4(float x, float y, float z, float w); 
#line 149
static __inline longlong1 make_longlong1(__int64 x); 
#line 151
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x); 
#line 153
static __inline longlong2 make_longlong2(__int64 x, __int64 y); 
#line 155
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y); 
#line 157
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z); 
#line 159
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z); 
#line 161
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w); 
#line 163
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w); 
#line 165
static __inline double1 make_double1(double x); 
#line 167
static __inline double2 make_double2(double x, double y); 
#line 169
static __inline double3 make_double3(double x, double y, double z); 
#line 171
static __inline double4 make_double4(double x, double y, double z, double w); 
#line 73 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\vector_functions.hpp"
static __inline char1 make_char1(signed char x) 
#line 74
{ 
#line 75
char1 t; (t.x) = x; return t; 
#line 76
} 
#line 78
static __inline uchar1 make_uchar1(unsigned char x) 
#line 79
{ 
#line 80
uchar1 t; (t.x) = x; return t; 
#line 81
} 
#line 83
static __inline char2 make_char2(signed char x, signed char y) 
#line 84
{ 
#line 85
char2 t; (t.x) = x; (t.y) = y; return t; 
#line 86
} 
#line 88
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
#line 89
{ 
#line 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
#line 91
} 
#line 93
static __inline char3 make_char3(signed char x, signed char y, signed char z) 
#line 94
{ 
#line 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 96
} 
#line 98
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
#line 99
{ 
#line 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 101
} 
#line 103
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
#line 104
{ 
#line 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 106
} 
#line 108
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
#line 109
{ 
#line 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 111
} 
#line 113
static __inline short1 make_short1(short x) 
#line 114
{ 
#line 115
short1 t; (t.x) = x; return t; 
#line 116
} 
#line 118
static __inline ushort1 make_ushort1(unsigned short x) 
#line 119
{ 
#line 120
ushort1 t; (t.x) = x; return t; 
#line 121
} 
#line 123
static __inline short2 make_short2(short x, short y) 
#line 124
{ 
#line 125
short2 t; (t.x) = x; (t.y) = y; return t; 
#line 126
} 
#line 128
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
#line 129
{ 
#line 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
#line 131
} 
#line 133
static __inline short3 make_short3(short x, short y, short z) 
#line 134
{ 
#line 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 136
} 
#line 138
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
#line 139
{ 
#line 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 141
} 
#line 143
static __inline short4 make_short4(short x, short y, short z, short w) 
#line 144
{ 
#line 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 146
} 
#line 148
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
#line 149
{ 
#line 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 151
} 
#line 153
static __inline int1 make_int1(int x) 
#line 154
{ 
#line 155
int1 t; (t.x) = x; return t; 
#line 156
} 
#line 158
static __inline uint1 make_uint1(unsigned x) 
#line 159
{ 
#line 160
uint1 t; (t.x) = x; return t; 
#line 161
} 
#line 163
static __inline int2 make_int2(int x, int y) 
#line 164
{ 
#line 165
int2 t; (t.x) = x; (t.y) = y; return t; 
#line 166
} 
#line 168
static __inline uint2 make_uint2(unsigned x, unsigned y) 
#line 169
{ 
#line 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
#line 171
} 
#line 173
static __inline int3 make_int3(int x, int y, int z) 
#line 174
{ 
#line 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 176
} 
#line 178
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
#line 179
{ 
#line 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 181
} 
#line 183
static __inline int4 make_int4(int x, int y, int z, int w) 
#line 184
{ 
#line 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 186
} 
#line 188
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
#line 189
{ 
#line 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 191
} 
#line 193
static __inline long1 make_long1(long x) 
#line 194
{ 
#line 195
long1 t; (t.x) = x; return t; 
#line 196
} 
#line 198
static __inline ulong1 make_ulong1(unsigned long x) 
#line 199
{ 
#line 200
ulong1 t; (t.x) = x; return t; 
#line 201
} 
#line 203
static __inline long2 make_long2(long x, long y) 
#line 204
{ 
#line 205
long2 t; (t.x) = x; (t.y) = y; return t; 
#line 206
} 
#line 208
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
#line 209
{ 
#line 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
#line 211
} 
#line 213
static __inline long3 make_long3(long x, long y, long z) 
#line 214
{ 
#line 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 216
} 
#line 218
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
#line 219
{ 
#line 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 221
} 
#line 223
static __inline long4 make_long4(long x, long y, long z, long w) 
#line 224
{ 
#line 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 226
} 
#line 228
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
#line 229
{ 
#line 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 231
} 
#line 233
static __inline float1 make_float1(float x) 
#line 234
{ 
#line 235
float1 t; (t.x) = x; return t; 
#line 236
} 
#line 238
static __inline float2 make_float2(float x, float y) 
#line 239
{ 
#line 240
float2 t; (t.x) = x; (t.y) = y; return t; 
#line 241
} 
#line 243
static __inline float3 make_float3(float x, float y, float z) 
#line 244
{ 
#line 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 246
} 
#line 248
static __inline float4 make_float4(float x, float y, float z, float w) 
#line 249
{ 
#line 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 251
} 
#line 253
static __inline longlong1 make_longlong1(__int64 x) 
#line 254
{ 
#line 255
longlong1 t; (t.x) = x; return t; 
#line 256
} 
#line 258
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x) 
#line 259
{ 
#line 260
ulonglong1 t; (t.x) = x; return t; 
#line 261
} 
#line 263
static __inline longlong2 make_longlong2(__int64 x, __int64 y) 
#line 264
{ 
#line 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
#line 266
} 
#line 268
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y) 
#line 269
{ 
#line 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
#line 271
} 
#line 273
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z) 
#line 274
{ 
#line 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 276
} 
#line 278
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z) 
#line 279
{ 
#line 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 281
} 
#line 283
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w) 
#line 284
{ 
#line 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 286
} 
#line 288
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w) 
#line 289
{ 
#line 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 291
} 
#line 293
static __inline double1 make_double1(double x) 
#line 294
{ 
#line 295
double1 t; (t.x) = x; return t; 
#line 296
} 
#line 298
static __inline double2 make_double2(double x, double y) 
#line 299
{ 
#line 300
double2 t; (t.x) = x; (t.y) = y; return t; 
#line 301
} 
#line 303
static __inline double3 make_double3(double x, double y, double z) 
#line 304
{ 
#line 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 306
} 
#line 308
static __inline double4 make_double4(double x, double y, double z, double w) 
#line 309
{ 
#line 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 311
} 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\errno.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
int *__cdecl _errno(); 
#line 26
errno_t __cdecl _set_errno(int _Value); 
#line 27
errno_t __cdecl _get_errno(int * _Value); 
#line 29
unsigned long *__cdecl __doserrno(); 
#line 32
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 33
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 134 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\errno.h"
}__pragma( pack ( pop )) 
#line 136
#pragma warning(pop)
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_string.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 21
[[nodiscard]] const void *__cdecl 
#line 22
memchr(const void * _Buf, int _Val, size_t _MaxCount); 
#line 28
[[nodiscard]] int __cdecl 
#line 29
memcmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 43 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_string.h"
void *__cdecl memcpy(void * _Dst, const void * _Src, size_t _Size); 
#line 50
void *__cdecl memmove(void * _Dst, const void * _Src, size_t _Size); 
#line 63 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_string.h"
void *__cdecl memset(void * _Dst, int _Val, size_t _Size); 
#line 69
[[nodiscard]] const char *__cdecl 
#line 70
strchr(const char * _Str, int _Val); 
#line 75
[[nodiscard]] const char *__cdecl 
#line 76
strrchr(const char * _Str, int _Ch); 
#line 81
[[nodiscard]] const char *__cdecl 
#line 82
strstr(const char * _Str, const char * _SubStr); 
#line 87
[[nodiscard]] const __wchar_t *__cdecl 
#line 89
wcschr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 94
[[nodiscard]] const __wchar_t *__cdecl 
#line 95
wcsrchr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 100
[[nodiscard]] const __wchar_t *__cdecl 
#line 102
wcsstr(const __wchar_t * _Str, const __wchar_t * _SubStr); 
#line 109
}__pragma( pack ( pop )) 
#line 113 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_string.h"
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_memcpy_s.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 39 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_memcpy_s.h"
static __inline errno_t __cdecl memcpy_s(void *const 
#line 40
_Destination, const rsize_t 
#line 41
_DestinationSize, const void *const 
#line 42
_Source, const rsize_t 
#line 43
_SourceSize) 
#line 45
{ 
#line 46
if (_SourceSize == (0)) 
#line 47
{ 
#line 48
return 0; 
#line 49
}  
#line 51
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 52
if ((_Source == (0)) || (_DestinationSize < _SourceSize)) 
#line 53
{ 
#line 54
memset(_Destination, 0, _DestinationSize); 
#line 56
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 57
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 60
return 22; 
#line 61
}  
#line 62
memcpy(_Destination, _Source, _SourceSize); 
#line 63
return 0; 
#line 64
} 
#line 67
static __inline errno_t __cdecl memmove_s(void *const 
#line 68
_Destination, const rsize_t 
#line 69
_DestinationSize, const void *const 
#line 70
_Source, const rsize_t 
#line 71
_SourceSize) 
#line 73
{ 
#line 74
if (_SourceSize == (0)) 
#line 75
{ 
#line 76
return 0; 
#line 77
}  
#line 79
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 80
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 81
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 83
memmove(_Destination, _Source, _SourceSize); 
#line 84
return 0; 
#line 85
} 
#line 93 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_memcpy_s.h"
}
#line 92
#pragma warning(pop)
__pragma( pack ( pop )) 
#line 17 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_memory.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 23
__pragma( pack ( push, 8 )) extern "C" {
#line 28
int __cdecl _memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 35
int __cdecl _memicmp_l(const void * _Buf1, const void * _Buf2, size_t _Size, _locale_t _Locale); 
#line 83 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_memory.h"
void *__cdecl memccpy(void * _Dst, const void * _Src, int _Val, size_t _Size); 
#line 91
int __cdecl memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 104 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_memory.h"
extern "C++" inline void *__cdecl memchr(void *
#line 105
_Pv, int 
#line 106
_C, size_t 
#line 107
_N) 
#line 109
{ 
#line 110
const void *const _Pvc = _Pv; 
#line 111
return const_cast< void *>(memchr(_Pvc, _C, _N)); 
#line 112
} 
#line 118 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_memory.h"
}__pragma( pack ( pop )) 
#line 122 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_memory.h"
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 20
__pragma( pack ( push, 8 )) extern "C" {
#line 32
errno_t __cdecl wcscat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 39
errno_t __cdecl wcscpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 46
errno_t __cdecl wcsncat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 54
errno_t __cdecl wcsncpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 62
__wchar_t *__cdecl wcstok_s(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 83 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__declspec(allocator) __wchar_t *__cdecl _wcsdup(const __wchar_t * _String); 
#line 93 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscat_s ( _Destination, _Size, _Source ); }}
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcscat(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 108 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcscmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 113
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscpy_s ( _Destination, _Size, _Source ); }}
#line 119 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcscpy(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 126 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcscspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 132
size_t __cdecl wcslen(const __wchar_t * _String); 
#line 145 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsnlen(const __wchar_t * _Source, size_t _MaxCount); 
#line 161 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
static __inline size_t __cdecl wcsnlen_s(const __wchar_t *
#line 162
_Source, size_t 
#line 163
_MaxCount) 
#line 165
{ 
#line 166
return (_Source == (0)) ? 0 : wcsnlen(_Source, _MaxCount); 
#line 167
} 
#line 171 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 178 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncat(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 187 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsncmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 193
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 200 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncpy(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 209 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
const __wchar_t *__cdecl wcspbrk(const __wchar_t * _String, const __wchar_t * _Control); 
#line 215
size_t __cdecl wcsspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 221
__wchar_t *__cdecl wcstok(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 239 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
static __inline __wchar_t *__cdecl _wcstok(__wchar_t *const 
#line 240
_String, const __wchar_t *const 
#line 241
_Delimiter) 
#line 243
{ 
#line 244
return wcstok(_String, _Delimiter, 0); 
#line 245
} 
#line 253 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
extern "C++" 
#line 252
__declspec(deprecated("wcstok has been changed to conform with the ISO C standard, adding an extra context parameter. To use the legacy Microsoft wcsto" "k, define _CRT_NON_CONFORMING_WCSTOK.")) inline __wchar_t *__cdecl 
#line 253
wcstok(__wchar_t *
#line 254
_String, const __wchar_t *
#line 255
_Delimiter) throw() 
#line 257
{ 
#line 258
return wcstok(_String, _Delimiter, 0); 
#line 259
} 
#line 268 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcserror(int _ErrorNumber); 
#line 273
errno_t __cdecl _wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, int _ErrorNumber); 
#line 279
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], int _Error ) throw ( ) { return _wcserror_s ( _Buffer, _Size, _Error ); }}
#line 288 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl __wcserror(const __wchar_t * _String); 
#line 292
errno_t __cdecl __wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _ErrorMessage); 
#line 298
extern "C++" {template < size_t _Size > inline errno_t __cdecl __wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _ErrorMessage ) throw ( ) { return __wcserror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 304 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
int __cdecl _wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 309
int __cdecl _wcsicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 315
int __cdecl _wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 321
int __cdecl _wcsnicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 328
errno_t __cdecl _wcsnset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value, size_t _MaxCount); 
#line 335
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsnset_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t _Value, size_t _MaxCount ) throw ( ) { return _wcsnset_s ( _Destination, _Size, _Value, _MaxCount ); }}
#line 342 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 350 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsrev(__wchar_t * _String); 
#line 354
errno_t __cdecl _wcsset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value); 
#line 360
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsset_s ( wchar_t ( & _String ) [ _Size ], wchar_t _Value ) throw ( ) { return _wcsset_s ( _String, _Size, _Value ); }}
#line 366 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 373 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s(__wchar_t * _String, size_t _SizeInWords); 
#line 378
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcslwr_s ( _String, _Size ); }}
#line 383 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr(__wchar_t * _String); 
#line 389 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s_l(__wchar_t * _String, size_t _SizeInWords, _locale_t _Locale); 
#line 395
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcslwr_s_l ( _String, _Size, _Locale ); }}
#line 401 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr_l(__wchar_t * _String, _locale_t _Locale); 
#line 409 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s(__wchar_t * _String, size_t _Size); 
#line 414
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcsupr_s ( _String, _Size ); }}
#line 419 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr(__wchar_t * _String); 
#line 425 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s_l(__wchar_t * _String, size_t _Size, _locale_t _Locale); 
#line 431
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcsupr_s_l ( _String, _Size, _Locale ); }}
#line 437 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr_l(__wchar_t * _String, _locale_t _Locale); 
#line 446 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsxfrm(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount); 
#line 454
size_t __cdecl _wcsxfrm_l(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 462
int __cdecl wcscoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 468
int __cdecl _wcscoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 475
int __cdecl _wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 481
int __cdecl _wcsicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 488
int __cdecl _wcsncoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 495
int __cdecl _wcsncoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 503
int __cdecl _wcsnicoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 510
int __cdecl _wcsnicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 525
extern "C++" {
#line 529
inline __wchar_t *__cdecl wcschr(__wchar_t *_String, __wchar_t _C) 
#line 530
{ 
#line 531
return const_cast< __wchar_t *>(wcschr(static_cast< const __wchar_t *>(_String), _C)); 
#line 532
} 
#line 535
inline __wchar_t *__cdecl wcspbrk(__wchar_t *_String, const __wchar_t *_Control) 
#line 536
{ 
#line 537
return const_cast< __wchar_t *>(wcspbrk(static_cast< const __wchar_t *>(_String), _Control)); 
#line 538
} 
#line 541
inline __wchar_t *__cdecl wcsrchr(__wchar_t *_String, __wchar_t _C) 
#line 542
{ 
#line 543
return const_cast< __wchar_t *>(wcsrchr(static_cast< const __wchar_t *>(_String), _C)); 
#line 544
} 
#line 548
inline __wchar_t *__cdecl wcsstr(__wchar_t *_String, const __wchar_t *_SubStr) 
#line 549
{ 
#line 550
return const_cast< __wchar_t *>(wcsstr(static_cast< const __wchar_t *>(_String), _SubStr)); 
#line 551
} 
#line 553
}
#line 570 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsdup(const __wchar_t * _String); 
#line 582 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 588
int __cdecl wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 596
__wchar_t *__cdecl wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 604
__wchar_t *__cdecl wcsrev(__wchar_t * _String); 
#line 610
__wchar_t *__cdecl wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 617
__wchar_t *__cdecl wcslwr(__wchar_t * _String); 
#line 623
__wchar_t *__cdecl wcsupr(__wchar_t * _String); 
#line 628
int __cdecl wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 637 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
}__pragma( pack ( pop )) 
#line 641 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 19 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 23
__pragma( pack ( push, 8 )) extern "C" {
#line 32
errno_t __cdecl strcpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 39
errno_t __cdecl strcat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 46
errno_t __cdecl strerror_s(char * _Buffer, size_t _SizeInBytes, int _ErrorNumber); 
#line 52
errno_t __cdecl strncat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 60
errno_t __cdecl strncpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 68
char *__cdecl strtok_s(char * _String, const char * _Delimiter, char ** _Context); 
#line 76 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
void *__cdecl _memccpy(void * _Dst, const void * _Src, int _Val, size_t _MaxCount); 
#line 83
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcat_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcat_s ( _Destination, _Size, _Source ); }}
#line 91 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl strcat(char * _Destination, const char * _Source); 
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
int __cdecl strcmp(const char * _Str1, const char * _Str2); 
#line 106
int __cdecl _strcmpi(const char * _String1, const char * _String2); 
#line 112
int __cdecl strcoll(const char * _String1, const char * _String2); 
#line 118
int __cdecl _strcoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 124
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcpy_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcpy_s ( _Destination, _Size, _Source ); }}
#line 130 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl strcpy(char * _Destination, const char * _Source); 
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
size_t __cdecl strcspn(const char * _Str, const char * _Control); 
#line 148 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
__declspec(allocator) char *__cdecl _strdup(const char * _Source); 
#line 159 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl _strerror(const char * _ErrorMessage); 
#line 164
errno_t __cdecl _strerror_s(char * _Buffer, size_t _SizeInBytes, const char * _ErrorMessage); 
#line 170
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strerror_s ( char ( & _Buffer ) [ _Size ], char const * _ErrorMessage ) throw ( ) { return _strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 178 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl strerror(int _ErrorMessage); 
#line 182
extern "C++" {template < size_t _Size > inline errno_t __cdecl strerror_s ( char ( & _Buffer ) [ _Size ], int _ErrorMessage ) throw ( ) { return strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 189 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
int __cdecl _stricmp(const char * _String1, const char * _String2); 
#line 195
int __cdecl _stricoll(const char * _String1, const char * _String2); 
#line 201
int __cdecl _stricoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 208
int __cdecl _stricmp_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 215
size_t __cdecl strlen(const char * _Str); 
#line 220
errno_t __cdecl _strlwr_s(char * _String, size_t _Size); 
#line 225
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strlwr_s ( _String, _Size ); }}
#line 230 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl _strlwr(char * _String); 
#line 236 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
errno_t __cdecl _strlwr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 242
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strlwr_s_l ( _String, _Size, _Locale ); }}
#line 248 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl _strlwr_l(char * _String, _locale_t _Locale); 
#line 255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncat_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 262 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl strncat(char * _Destination, const char * _Source, size_t _Count); 
#line 271 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
int __cdecl strncmp(const char * _Str1, const char * _Str2, size_t _MaxCount); 
#line 278
int __cdecl _strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 285
int __cdecl _strnicmp_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 293
int __cdecl _strnicoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 300
int __cdecl _strnicoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 308
int __cdecl _strncoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 315
int __cdecl _strncoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 322
size_t __cdecl __strncnt(const char * _String, size_t _Count); 
#line 327
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncpy_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 334 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl strncpy(char * _Destination, const char * _Source, size_t _Count); 
#line 351 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
size_t __cdecl strnlen(const char * _String, size_t _MaxCount); 
#line 367 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
static __inline size_t __cdecl strnlen_s(const char *
#line 368
_String, size_t 
#line 369
_MaxCount) 
#line 371
{ 
#line 372
return (_String == (0)) ? 0 : strnlen(_String, _MaxCount); 
#line 373
} 
#line 378 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
errno_t __cdecl _strnset_s(char * _String, size_t _SizeInBytes, int _Value, size_t _MaxCount); 
#line 385
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strnset_s ( char ( & _Destination ) [ _Size ], int _Value, size_t _Count ) throw ( ) { return _strnset_s ( _Destination, _Size, _Value, _Count ); }}
#line 392 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl _strnset(char * _Destination, int _Value, size_t _Count); 
#line 401 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
const char *__cdecl strpbrk(const char * _Str, const char * _Control); 
#line 406
char *__cdecl _strrev(char * _Str); 
#line 411
errno_t __cdecl _strset_s(char * _Destination, size_t _DestinationSize, int _Value); 
#line 417
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strset_s ( char ( & _Destination ) [ _Size ], int _Value ) throw ( ) { return _strset_s ( _Destination, _Size, _Value ); }}
#line 423 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl _strset(char * _Destination, int _Value); 
#line 430 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
size_t __cdecl strspn(const char * _Str, const char * _Control); 
#line 436
char *__cdecl strtok(char * _String, const char * _Delimiter); 
#line 442
errno_t __cdecl _strupr_s(char * _String, size_t _Size); 
#line 447
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strupr_s ( _String, _Size ); }}
#line 452 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl _strupr(char * _String); 
#line 458 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
errno_t __cdecl _strupr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 464
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strupr_s_l ( _String, _Size, _Locale ); }}
#line 470 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl _strupr_l(char * _String, _locale_t _Locale); 
#line 479 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
size_t __cdecl strxfrm(char * _Destination, const char * _Source, size_t _MaxCount); 
#line 487
size_t __cdecl _strxfrm_l(char * _Destination, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 497
extern "C++" {
#line 500
inline char *__cdecl strchr(char *const _String, const int _Ch) 
#line 501
{ 
#line 502
return const_cast< char *>(strchr(static_cast< const char *>(_String), _Ch)); 
#line 503
} 
#line 506
inline char *__cdecl strpbrk(char *const _String, const char *const _Control) 
#line 507
{ 
#line 508
return const_cast< char *>(strpbrk(static_cast< const char *>(_String), _Control)); 
#line 509
} 
#line 512
inline char *__cdecl strrchr(char *const _String, const int _Ch) 
#line 513
{ 
#line 514
return const_cast< char *>(strrchr(static_cast< const char *>(_String), _Ch)); 
#line 515
} 
#line 518
inline char *__cdecl strstr(char *const _String, const char *const _SubString) 
#line 519
{ 
#line 520
return const_cast< char *>(strstr(static_cast< const char *>(_String), _SubString)); 
#line 521
} 
#line 522
}
#line 532 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
char *__cdecl strdup(const char * _String); 
#line 539
int __cdecl strcmpi(const char * _String1, const char * _String2); 
#line 545
int __cdecl stricmp(const char * _String1, const char * _String2); 
#line 551
char *__cdecl strlwr(char * _String); 
#line 556
int __cdecl strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 563
char *__cdecl strnset(char * _String, int _Value, size_t _MaxCount); 
#line 570
char *__cdecl strrev(char * _String); 
#line 575
char *__cdecl strset(char * _String, int _Value); 
#line 580
char *__cdecl strupr(char * _String); 
#line 588 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\string.h"
}__pragma( pack ( pop )) 
#line 590
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 26
struct tm { 
#line 28
int tm_sec; 
#line 29
int tm_min; 
#line 30
int tm_hour; 
#line 31
int tm_mday; 
#line 32
int tm_mon; 
#line 33
int tm_year; 
#line 34
int tm_wday; 
#line 35
int tm_yday; 
#line 36
int tm_isdst; 
#line 37
}; 
#line 48
__wchar_t *__cdecl _wasctime(const tm * _Tm); 
#line 54
errno_t __cdecl _wasctime_s(__wchar_t * _Buffer, size_t _SizeInWords, const tm * _Tm); 
#line 60
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wasctime_s ( wchar_t ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return _wasctime_s ( _Buffer, _Size, _Time ); }}
#line 69 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
size_t __cdecl wcsftime(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm); 
#line 78
size_t __cdecl _wcsftime_l(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm, _locale_t _Locale); 
#line 88
__wchar_t *__cdecl _wctime32(const __time32_t * _Time); 
#line 93
errno_t __cdecl _wctime32_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time32_t * _Time); 
#line 99
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime32_s ( wchar_t ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _wctime32_s ( _Buffer, _Size, _Time ); }}
#line 108 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wctime64(const __time64_t * _Time); 
#line 113
errno_t __cdecl _wctime64_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time64_t * _Time); 
#line 118
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime64_s ( wchar_t ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _wctime64_s ( _Buffer, _Size, _Time ); }}
#line 125 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrdate_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 130
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrdate_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrdate_s ( _Buffer, _Size ); }}
#line 135 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrdate(__wchar_t * _Buffer); 
#line 141 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrtime_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 146
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrtime_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrtime_s ( _Buffer, _Size ); }}
#line 151 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrtime(__wchar_t * _Buffer); 
#line 186 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
static __inline __wchar_t *__cdecl _wctime(const time_t *const 
#line 187
_Time) 
#line 188
{ 
#line 189
return _wctime64(_Time); 
#line 190
} 
#line 193
static __inline errno_t __cdecl _wctime_s(__wchar_t *const 
#line 194
_Buffer, const size_t 
#line 195
_SizeInWords, const time_t *const 
#line 196
_Time) 
#line 198
{ 
#line 199
return _wctime64_s(_Buffer, _SizeInWords, _Time); 
#line 200
} 
#line 205 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_wtime.h"
}__pragma( pack ( pop )) 
#line 207
#pragma warning(pop)
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 19
__pragma( pack ( push, 8 )) extern "C" {
#line 30 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
typedef long clock_t; 
#line 32
struct _timespec32 { 
#line 34
__time32_t tv_sec; 
#line 35
long tv_nsec; 
#line 36
}; 
#line 38
struct _timespec64 { 
#line 40
__time64_t tv_sec; 
#line 41
long tv_nsec; 
#line 42
}; 
#line 45
struct timespec { 
#line 47
time_t tv_sec; 
#line 48
long tv_nsec; 
#line 49
}; 
#line 68 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
int *__cdecl __daylight(); 
#line 74
long *__cdecl __dstbias(); 
#line 80
long *__cdecl __timezone(); 
#line 86
char **__cdecl __tzname(); 
#line 91
errno_t __cdecl _get_daylight(int * _Daylight); 
#line 96
errno_t __cdecl _get_dstbias(long * _DaylightSavingsBias); 
#line 101
errno_t __cdecl _get_timezone(long * _TimeZone); 
#line 106
errno_t __cdecl _get_tzname(size_t * _ReturnValue, char * _Buffer, size_t _SizeInBytes, int _Index); 
#line 123
char *__cdecl asctime(const tm * _Tm); 
#line 130
errno_t __cdecl asctime_s(char * _Buffer, size_t _SizeInBytes, const tm * _Tm); 
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl asctime_s ( char ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return asctime_s ( _Buffer, _Size, _Time ); }}
#line 144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
clock_t __cdecl clock(); 
#line 149
char *__cdecl _ctime32(const __time32_t * _Time); 
#line 154
errno_t __cdecl _ctime32_s(char * _Buffer, size_t _SizeInBytes, const __time32_t * _Time); 
#line 160
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime32_s ( char ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _ctime32_s ( _Buffer, _Size, _Time ); }}
#line 169 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
char *__cdecl _ctime64(const __time64_t * _Time); 
#line 174
errno_t __cdecl _ctime64_s(char * _Buffer, size_t _SizeInBytes, const __time64_t * _Time); 
#line 180
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime64_s ( char ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _ctime64_s ( _Buffer, _Size, _Time ); }}
#line 187 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
double __cdecl _difftime32(__time32_t _Time1, __time32_t _Time2); 
#line 193
double __cdecl _difftime64(__time64_t _Time1, __time64_t _Time2); 
#line 200
tm *__cdecl _gmtime32(const __time32_t * _Time); 
#line 205
errno_t __cdecl _gmtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 212
tm *__cdecl _gmtime64(const __time64_t * _Time); 
#line 217
errno_t __cdecl _gmtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 224
tm *__cdecl _localtime32(const __time32_t * _Time); 
#line 229
errno_t __cdecl _localtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 236
tm *__cdecl _localtime64(const __time64_t * _Time); 
#line 241
errno_t __cdecl _localtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 247
__time32_t __cdecl _mkgmtime32(tm * _Tm); 
#line 252
__time64_t __cdecl _mkgmtime64(tm * _Tm); 
#line 257
__time32_t __cdecl _mktime32(tm * _Tm); 
#line 262
__time64_t __cdecl _mktime64(tm * _Tm); 
#line 268
size_t __cdecl strftime(char * _Buffer, size_t _SizeInBytes, const char * _Format, const tm * _Tm); 
#line 277
size_t __cdecl _strftime_l(char * _Buffer, size_t _MaxSize, const char * _Format, const tm * _Tm, _locale_t _Locale); 
#line 286
errno_t __cdecl _strdate_s(char * _Buffer, size_t _SizeInBytes); 
#line 291
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strdate_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strdate_s ( _Buffer, _Size ); }}
#line 296 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
char *__cdecl _strdate(char * _Buffer); 
#line 302 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
errno_t __cdecl _strtime_s(char * _Buffer, size_t _SizeInBytes); 
#line 307
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strtime_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strtime_s ( _Buffer, _Size ); }}
#line 312 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
char *__cdecl _strtime(char * _Buffer); 
#line 317 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
__time32_t __cdecl _time32(__time32_t * _Time); 
#line 321
__time64_t __cdecl _time64(__time64_t * _Time); 
#line 327
int __cdecl _timespec32_get(_timespec32 * _Ts, int _Base); 
#line 334
int __cdecl _timespec64_get(_timespec64 * _Ts, int _Base); 
#line 348
void __cdecl _tzset(); 
#line 351
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using GetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 352
_getsystime(tm * _Tm); 
#line 356
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 357
_setsystime(tm * _Tm, unsigned _Milliseconds); 
#line 501 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
static __inline char *__cdecl ctime(const time_t *const 
#line 502
_Time) 
#line 504
{ 
#line 505
return _ctime64(_Time); 
#line 506
} 
#line 509
static __inline double __cdecl difftime(const time_t 
#line 510
_Time1, const time_t 
#line 511
_Time2) 
#line 513
{ 
#line 514
return _difftime64(_Time1, _Time2); 
#line 515
} 
#line 518
static __inline tm *__cdecl gmtime(const time_t *const 
#line 519
_Time) 
#line 520
{ 
#line 521
return _gmtime64(_Time); 
#line 522
} 
#line 525
static __inline tm *__cdecl localtime(const time_t *const 
#line 526
_Time) 
#line 528
{ 
#line 529
return _localtime64(_Time); 
#line 530
} 
#line 533
static __inline time_t __cdecl _mkgmtime(tm *const 
#line 534
_Tm) 
#line 536
{ 
#line 537
return _mkgmtime64(_Tm); 
#line 538
} 
#line 541
static __inline time_t __cdecl mktime(tm *const 
#line 542
_Tm) 
#line 544
{ 
#line 545
return _mktime64(_Tm); 
#line 546
} 
#line 548
static __inline time_t __cdecl time(time_t *const 
#line 549
_Time) 
#line 551
{ 
#line 552
return _time64(_Time); 
#line 553
} 
#line 556
static __inline int __cdecl timespec_get(timespec *const 
#line 557
_Ts, const int 
#line 558
_Base) 
#line 560
{ 
#line 561
return _timespec64_get((_timespec64 *)_Ts, _Base); 
#line 562
} 
#line 566
static __inline errno_t __cdecl ctime_s(char *const 
#line 567
_Buffer, const size_t 
#line 568
_SizeInBytes, const time_t *const 
#line 569
_Time) 
#line 571
{ 
#line 572
return _ctime64_s(_Buffer, _SizeInBytes, _Time); 
#line 573
} 
#line 603 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
static __inline errno_t __cdecl gmtime_s(tm *const 
#line 604
_Tm, const time_t *const 
#line 605
_Time) 
#line 607
{ 
#line 608
return _gmtime64_s(_Tm, _Time); 
#line 609
} 
#line 612
static __inline errno_t __cdecl localtime_s(tm *const 
#line 613
_Tm, const time_t *const 
#line 614
_Time) 
#line 616
{ 
#line 617
return _localtime64_s(_Tm, _Time); 
#line 618
} 
#line 638 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
void __cdecl tzset(); 
#line 645 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\time.h"
}__pragma( pack ( pop )) 
#line 647
#pragma warning(pop)
#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt/common_functions.h"
extern "C" {
#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt/common_functions.h"
extern clock_t __cdecl clock(); 
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt/common_functions.h"
extern void *__cdecl memset(void *, int, size_t); 
#line 97
extern void *__cdecl memcpy(void *, const void *, size_t); 
#line 99
}
#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern "C" {
#line 233 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern int __cdecl abs(int a); 
#line 244
extern long __cdecl labs(long a); 
#line 255
extern __int64 llabs(__int64 a); 
#line 283 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl fabs(double x); 
#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern __inline float fabsf(float x); 
#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline int min(const int a, const int b); 
#line 324
extern inline unsigned umin(const unsigned a, const unsigned b); 
#line 331
extern inline __int64 llmin(const __int64 a, const __int64 b); 
#line 338
extern inline unsigned __int64 ullmin(const unsigned __int64 a, const unsigned __int64 b); 
#line 361 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl fminf(float x, float y); 
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl fmin(double x, double y); 
#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline int max(const int a, const int b); 
#line 400
extern inline unsigned umax(const unsigned a, const unsigned b); 
#line 407
extern inline __int64 llmax(const __int64 a, const __int64 b); 
#line 414
extern inline unsigned __int64 ullmax(const unsigned __int64 a, const unsigned __int64 b); 
#line 437 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl fmaxf(float x, float y); 
#line 457 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl fmax(double, double); 
#line 477 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl sin(double x); 
#line 495
extern double __cdecl cos(double x); 
#line 511 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern void sincos(double x, double * sptr, double * cptr); 
#line 524
extern void sincosf(float x, float * sptr, float * cptr); 
#line 547 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl tan(double x); 
#line 571
extern double __cdecl sqrt(double x); 
#line 597 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double rsqrt(double x); 
#line 621
extern float rsqrtf(float x); 
#line 650 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl log2(double x); 
#line 679 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl exp2(double x); 
#line 708 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl exp2f(float x); 
#line 737 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double exp10(double x); 
#line 764
extern float exp10f(float x); 
#line 800 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl expm1(double x); 
#line 833 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl expm1f(float x); 
#line 860 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl log2f(float x); 
#line 883 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl log10(double x); 
#line 909
extern double __cdecl log(double x); 
#line 938 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl log1p(double x); 
#line 968 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl log1pf(float x); 
#line 992 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl floor(double x); 
#line 1021
extern double __cdecl exp(double x); 
#line 1040
extern double __cdecl cosh(double x); 
#line 1060
extern double __cdecl sinh(double x); 
#line 1080
extern double __cdecl tanh(double x); 
#line 1106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl acosh(double x); 
#line 1133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl acoshf(float x); 
#line 1157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl asinh(double x); 
#line 1181 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl asinhf(float x); 
#line 1206 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl atanh(double x); 
#line 1231 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl atanhf(float x); 
#line 1247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl ldexp(double x, int exp); 
#line 1262
extern __inline float ldexpf(float x, int exp); 
#line 1285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl logb(double x); 
#line 1309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl logbf(float x); 
#line 1333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern int __cdecl ilogb(double x); 
#line 1357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern int __cdecl ilogbf(float x); 
#line 1385 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl scalbn(double x, int n); 
#line 1413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl scalbnf(float x, int n); 
#line 1441 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl scalbln(double x, long n); 
#line 1469 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl scalblnf(float x, long n); 
#line 1499 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl frexp(double x, int * nptr); 
#line 1528
extern __inline float frexpf(float x, int * nptr); 
#line 1553 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl round(double x); 
#line 1579 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl roundf(float x); 
#line 1597 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern long __cdecl lround(double x); 
#line 1615 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern long __cdecl lroundf(float x); 
#line 1633 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern __int64 __cdecl llround(double x); 
#line 1651 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern __int64 __cdecl llroundf(float x); 
#line 1721 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl rintf(float x); 
#line 1738 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern long __cdecl lrint(double x); 
#line 1755 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern long __cdecl lrintf(float x); 
#line 1772 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrint(double x); 
#line 1789 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrintf(float x); 
#line 1813 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl nearbyint(double x); 
#line 1837 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl nearbyintf(float x); 
#line 1859 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl ceil(double x); 
#line 1884 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl trunc(double x); 
#line 1910 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl truncf(float x); 
#line 1932 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl fdim(double x, double y); 
#line 1953 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl fdimf(float x, float y); 
#line 2034 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl atan2(double y, double x); 
#line 2060
extern double __cdecl atan(double x); 
#line 2077
extern double __cdecl acos(double x); 
#line 2099
extern double __cdecl asin(double x); 
#line 2127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl hypot(double x, double y); 
#line 2185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline float __cdecl hypotf(float x, float y); 
#line 2461 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl cbrt(double x); 
#line 2488 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl cbrtf(float x); 
#line 2512 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double rcbrt(double x); 
#line 2533
extern float rcbrtf(float x); 
#line 2556
extern double sinpi(double x); 
#line 2579
extern float sinpif(float x); 
#line 2601
extern double cospi(double x); 
#line 2623
extern float cospif(float x); 
#line 2636
extern void sincospi(double x, double * sptr, double * cptr); 
#line 2649
extern void sincospif(float x, float * sptr, float * cptr); 
#line 2735 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl pow(double x, double y); 
#line 2759
extern double __cdecl modf(double x, double * iptr); 
#line 2786
extern double __cdecl fmod(double x, double y); 
#line 2818 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl remainder(double x, double y); 
#line 2851 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl remainderf(float x, float y); 
#line 2889 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl remquo(double x, double y, int * quo); 
#line 2927 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl remquof(float x, float y, int * quo); 
#line 2946 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl j0(double x); 
#line 2968 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float j0f(float x); 
#line 2995 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl j1(double x); 
#line 3022 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float j1f(float x); 
#line 3045 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl jn(int n, double x); 
#line 3068 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float jnf(int n, float x); 
#line 3095 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl y0(double x); 
#line 3122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float y0f(float x); 
#line 3149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl y1(double x); 
#line 3176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float y1f(float x); 
#line 3204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl yn(int n, double x); 
#line 3232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float ynf(int n, float x); 
#line 3330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl erf(double x); 
#line 3355 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl erff(float x); 
#line 3383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double erfinv(double x); 
#line 3406
extern float erfinvf(float x); 
#line 3432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl erfc(double x); 
#line 3455 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl erfcf(float x); 
#line 3487 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl lgamma(double x); 
#line 3513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double erfcinv(double x); 
#line 3534
extern float erfcinvf(float x); 
#line 3556
extern double normcdfinv(double x); 
#line 3578
extern float normcdfinvf(float x); 
#line 3597
extern double normcdf(double x); 
#line 3616
extern float normcdff(float x); 
#line 3636
extern double erfcx(double x); 
#line 3656
extern float erfcxf(float x); 
#line 3691 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl lgammaf(float x); 
#line 3720 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl tgamma(double x); 
#line 3749 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl tgammaf(float x); 
#line 3763 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl copysign(double x, double y); 
#line 3777 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl copysignf(float x, float y); 
#line 3796 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl nextafter(double x, double y); 
#line 3815 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl nextafterf(float x, float y); 
#line 3831 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl nan(const char * tagp); 
#line 3847 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl nanf(const char * tagp); 
#line 3852 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern int __isinff(float); 
#line 3853
extern int __isnanf(float); 
#line 3863 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern int __finite(double); 
#line 3864
extern int __finitef(float); 
#line 3865
extern int __signbit(double); 
#line 3866
extern int __isnan(double); 
#line 3867
extern int __isinf(double); 
#line 3870 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern int __signbitf(float); 
#line 3923 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern double __cdecl fma(double x, double y, double z); 
#line 3973 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl fmaf(float x, float y, float z); 
#line 3982 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern int __signbitl(long double); 
#line 3988 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern int __finitel(long double); 
#line 3989
extern int __isinfl(long double); 
#line 3990
extern int __isnanl(long double); 
#line 3994 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern float __cdecl acosf(float); 
#line 3995
extern float __cdecl asinf(float); 
#line 3996
extern float __cdecl atanf(float); 
#line 3997
extern float __cdecl atan2f(float, float); 
#line 3998
extern float __cdecl cosf(float); 
#line 3999
extern float __cdecl sinf(float); 
#line 4000
extern float __cdecl tanf(float); 
#line 4001
extern float __cdecl coshf(float); 
#line 4002
extern float __cdecl sinhf(float); 
#line 4003
extern float __cdecl tanhf(float); 
#line 4004
extern float __cdecl expf(float); 
#line 4005
extern float __cdecl logf(float); 
#line 4006
extern float __cdecl log10f(float); 
#line 4007
extern float __cdecl modff(float, float *); 
#line 4008
extern float __cdecl powf(float, float); 
#line 4009
extern float __cdecl sqrtf(float); 
#line 4010
extern float __cdecl ceilf(float); 
#line 4011
extern float __cdecl floorf(float); 
#line 4012
extern float __cdecl fmodf(float, float); 
#line 4574 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
}
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
struct _exception { 
#line 25
int type; 
#line 26
char *name; 
#line 27
double arg1; 
#line 28
double arg2; 
#line 29
double retval; 
#line 30
}; 
#line 37
struct _complex { 
#line 39
double x, y; 
#line 40
}; 
#line 59 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
typedef float float_t; 
#line 60
typedef double double_t; 
#line 78 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
extern const double _HUGE; 
#line 175 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
void __cdecl _fperrraise(int _Except); 
#line 177
short __cdecl _dclass(double _X); 
#line 178
short __cdecl _ldclass(long double _X); 
#line 179
short __cdecl _fdclass(float _X); 
#line 181
int __cdecl _dsign(double _X); 
#line 182
int __cdecl _ldsign(long double _X); 
#line 183
int __cdecl _fdsign(float _X); 
#line 185
int __cdecl _dpcomp(double _X, double _Y); 
#line 186
int __cdecl _ldpcomp(long double _X, long double _Y); 
#line 187
int __cdecl _fdpcomp(float _X, float _Y); 
#line 189
short __cdecl _dtest(double * _Px); 
#line 190
short __cdecl _ldtest(long double * _Px); 
#line 191
short __cdecl _fdtest(float * _Px); 
#line 193
short __cdecl _d_int(double * _Px, short _Xexp); 
#line 194
short __cdecl _ld_int(long double * _Px, short _Xexp); 
#line 195
short __cdecl _fd_int(float * _Px, short _Xexp); 
#line 197
short __cdecl _dscale(double * _Px, long _Lexp); 
#line 198
short __cdecl _ldscale(long double * _Px, long _Lexp); 
#line 199
short __cdecl _fdscale(float * _Px, long _Lexp); 
#line 201
short __cdecl _dunscale(short * _Pex, double * _Px); 
#line 202
short __cdecl _ldunscale(short * _Pex, long double * _Px); 
#line 203
short __cdecl _fdunscale(short * _Pex, float * _Px); 
#line 205
short __cdecl _dexp(double * _Px, double _Y, long _Eoff); 
#line 206
short __cdecl _ldexp(long double * _Px, long double _Y, long _Eoff); 
#line 207
short __cdecl _fdexp(float * _Px, float _Y, long _Eoff); 
#line 209
short __cdecl _dnorm(unsigned short * _Ps); 
#line 210
short __cdecl _fdnorm(unsigned short * _Ps); 
#line 212
double __cdecl _dpoly(double _X, const double * _Tab, int _N); 
#line 213
long double __cdecl _ldpoly(long double _X, const long double * _Tab, int _N); 
#line 214
float __cdecl _fdpoly(float _X, const float * _Tab, int _N); 
#line 216
double __cdecl _dlog(double _X, int _Baseflag); 
#line 217
long double __cdecl _ldlog(long double _X, int _Baseflag); 
#line 218
float __cdecl _fdlog(float _X, int _Baseflag); 
#line 220
double __cdecl _dsin(double _X, unsigned _Qoff); 
#line 221
long double __cdecl _ldsin(long double _X, unsigned _Qoff); 
#line 222
float __cdecl _fdsin(float _X, unsigned _Qoff); 
#line 229
typedef 
#line 226
union { 
#line 227
unsigned short _Sh[4]; 
#line 228
double _Val; 
#line 229
} _double_val; 
#line 236
typedef 
#line 233
union { 
#line 234
unsigned short _Sh[2]; 
#line 235
float _Val; 
#line 236
} _float_val; 
#line 243
typedef 
#line 240
union { 
#line 241
unsigned short _Sh[4]; 
#line 242
long double _Val; 
#line 243
} _ldouble_val; 
#line 251
typedef 
#line 246
union { 
#line 247
unsigned short _Word[4]; 
#line 248
float _Float; 
#line 249
double _Double; 
#line 250
long double _Long_double; 
#line 251
} _float_const; 
#line 253
extern const _float_const _Denorm_C, _Inf_C, _Nan_C, _Snan_C, _Hugeval_C; 
#line 254
extern const _float_const _FDenorm_C, _FInf_C, _FNan_C, _FSnan_C; 
#line 255
extern const _float_const _LDenorm_C, _LInf_C, _LNan_C, _LSnan_C; 
#line 257
extern const _float_const _Eps_C, _Rteps_C; 
#line 258
extern const _float_const _FEps_C, _FRteps_C; 
#line 259
extern const _float_const _LEps_C, _LRteps_C; 
#line 261
extern const double _Zero_C, _Xbig_C; 
#line 262
extern const float _FZero_C, _FXbig_C; 
#line 263
extern const long double _LZero_C, _LXbig_C; 
#line 292
extern "C++" {
#line 294
inline int fpclassify(float _X) throw() 
#line 295
{ 
#line 296
return _fdtest(&_X); 
#line 297
} 
#line 299
inline int fpclassify(double _X) throw() 
#line 300
{ 
#line 301
return _dtest(&_X); 
#line 302
} 
#line 304
inline int fpclassify(long double _X) throw() 
#line 305
{ 
#line 306
return _ldtest(&_X); 
#line 307
} 
#line 309
inline bool signbit(float _X) throw() 
#line 310
{ 
#line 311
return _fdsign(_X) != 0; 
#line 312
} 
#line 314
inline bool signbit(double _X) throw() 
#line 315
{ 
#line 316
return _dsign(_X) != 0; 
#line 317
} 
#line 319
inline bool signbit(long double _X) throw() 
#line 320
{ 
#line 321
return _ldsign(_X) != 0; 
#line 322
} 
#line 324
inline int _fpcomp(float _X, float _Y) throw() 
#line 325
{ 
#line 326
return _fdpcomp(_X, _Y); 
#line 327
} 
#line 329
inline int _fpcomp(double _X, double _Y) throw() 
#line 330
{ 
#line 331
return _dpcomp(_X, _Y); 
#line 332
} 
#line 334
inline int _fpcomp(long double _X, long double _Y) throw() 
#line 335
{ 
#line 336
return _ldpcomp(_X, _Y); 
#line 337
} 
#line 339
template< class _Trc, class _Tre> struct _Combined_type { 
#line 341
typedef float _Type; 
#line 342
}; 
#line 344
template<> struct _Combined_type< float, double>  { 
#line 346
typedef double _Type; 
#line 347
}; 
#line 349
template<> struct _Combined_type< float, long double>  { 
#line 351
typedef long double _Type; 
#line 352
}; 
#line 354
template< class _Ty, class _T2> struct _Real_widened { 
#line 356
typedef long double _Type; 
#line 357
}; 
#line 359
template<> struct _Real_widened< float, float>  { 
#line 361
typedef float _Type; 
#line 362
}; 
#line 364
template<> struct _Real_widened< float, double>  { 
#line 366
typedef double _Type; 
#line 367
}; 
#line 369
template<> struct _Real_widened< double, float>  { 
#line 371
typedef double _Type; 
#line 372
}; 
#line 374
template<> struct _Real_widened< double, double>  { 
#line 376
typedef double _Type; 
#line 377
}; 
#line 379
template< class _Ty> struct _Real_type { 
#line 381
typedef double _Type; 
#line 382
}; 
#line 384
template<> struct _Real_type< float>  { 
#line 386
typedef float _Type; 
#line 387
}; 
#line 389
template<> struct _Real_type< long double>  { 
#line 391
typedef long double _Type; 
#line 392
}; 
#line 394
template < class _T1, class _T2 >
      inline int _fpcomp ( _T1 _X, _T2 _Y ) throw ( )
    {
        typedef typename _Combined_type < float,
            typename _Real_widened <
            typename _Real_type < _T1 > :: _Type,
            typename _Real_type < _T2 > :: _Type > :: _Type > :: _Type _Tw;
        return _fpcomp ( ( _Tw ) _X, ( _Tw ) _Y );
    }
#line 404
template < class _Ty >
      inline bool isfinite ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) <= 0;
    }
#line 410
template < class _Ty >
      inline bool isinf ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 1;
    }
#line 416
template < class _Ty >
      inline bool isnan ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 2;
    }
#line 422
template < class _Ty >
      inline bool isnormal ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == ( - 1 );
    }
#line 428
template < class _Ty1, class _Ty2 >
      inline bool isgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 4 ) != 0;
    }
#line 434
template < class _Ty1, class _Ty2 >
      inline bool isgreaterequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 2 | 4 ) ) != 0;
    }
#line 440
template < class _Ty1, class _Ty2 >
      inline bool isless ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 1 ) != 0;
    }
#line 446
template < class _Ty1, class _Ty2 >
      inline bool islessequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 2 ) ) != 0;
    }
#line 452
template < class _Ty1, class _Ty2 >
      inline bool islessgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 4 ) ) != 0;
    }
#line 458
template < class _Ty1, class _Ty2 >
      inline bool isunordered ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return _fpcomp ( _X, _Y ) == 0;
    }
#line 463
}
#line 470 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
int __cdecl abs(int _X); 
#line 471
long __cdecl labs(long _X); 
#line 472
__int64 __cdecl llabs(__int64 _X); 
#line 474
double __cdecl acos(double _X); 
#line 475
double __cdecl asin(double _X); 
#line 476
double __cdecl atan(double _X); 
#line 477
double __cdecl atan2(double _Y, double _X); 
#line 479
double __cdecl cos(double _X); 
#line 480
double __cdecl cosh(double _X); 
#line 481
double __cdecl exp(double _X); 
#line 482
double __cdecl fabs(double _X); 
#line 483
double __cdecl fmod(double _X, double _Y); 
#line 484
double __cdecl log(double _X); 
#line 485
double __cdecl log10(double _X); 
#line 486
double __cdecl pow(double _X, double _Y); 
#line 487
double __cdecl sin(double _X); 
#line 488
double __cdecl sinh(double _X); 
#line 489
double __cdecl sqrt(double _X); 
#line 490
double __cdecl tan(double _X); 
#line 491
double __cdecl tanh(double _X); 
#line 493
double __cdecl acosh(double _X); 
#line 494
double __cdecl asinh(double _X); 
#line 495
double __cdecl atanh(double _X); 
#line 496
double __cdecl atof(const char * _String); 
#line 497
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 498
double __cdecl _cabs(_complex _Complex_value); 
#line 499
double __cdecl cbrt(double _X); 
#line 500
double __cdecl ceil(double _X); 
#line 501
double __cdecl _chgsign(double _X); 
#line 502
double __cdecl copysign(double _Number, double _Sign); 
#line 503
double __cdecl _copysign(double _Number, double _Sign); 
#line 504
double __cdecl erf(double _X); 
#line 505
double __cdecl erfc(double _X); 
#line 506
double __cdecl exp2(double _X); 
#line 507
double __cdecl expm1(double _X); 
#line 508
double __cdecl fdim(double _X, double _Y); 
#line 509
double __cdecl floor(double _X); 
#line 510
double __cdecl fma(double _X, double _Y, double _Z); 
#line 511
double __cdecl fmax(double _X, double _Y); 
#line 512
double __cdecl fmin(double _X, double _Y); 
#line 513
double __cdecl frexp(double _X, int * _Y); 
#line 514
double __cdecl hypot(double _X, double _Y); 
#line 515
double __cdecl _hypot(double _X, double _Y); 
#line 516
int __cdecl ilogb(double _X); 
#line 517
double __cdecl ldexp(double _X, int _Y); 
#line 518
double __cdecl lgamma(double _X); 
#line 519
__int64 __cdecl llrint(double _X); 
#line 520
__int64 __cdecl llround(double _X); 
#line 521
double __cdecl log1p(double _X); 
#line 522
double __cdecl log2(double _X); 
#line 523
double __cdecl logb(double _X); 
#line 524
long __cdecl lrint(double _X); 
#line 525
long __cdecl lround(double _X); 
#line 527
int __cdecl _matherr(_exception * _Except); 
#line 529
double __cdecl modf(double _X, double * _Y); 
#line 530
double __cdecl nan(const char * _X); 
#line 531
double __cdecl nearbyint(double _X); 
#line 532
double __cdecl nextafter(double _X, double _Y); 
#line 533
double __cdecl nexttoward(double _X, long double _Y); 
#line 534
double __cdecl remainder(double _X, double _Y); 
#line 535
double __cdecl remquo(double _X, double _Y, int * _Z); 
#line 536
double __cdecl rint(double _X); 
#line 537
double __cdecl round(double _X); 
#line 538
double __cdecl scalbln(double _X, long _Y); 
#line 539
double __cdecl scalbn(double _X, int _Y); 
#line 540
double __cdecl tgamma(double _X); 
#line 541
double __cdecl trunc(double _X); 
#line 542
double __cdecl _j0(double _X); 
#line 543
double __cdecl _j1(double _X); 
#line 544
double __cdecl _jn(int _X, double _Y); 
#line 545
double __cdecl _y0(double _X); 
#line 546
double __cdecl _y1(double _X); 
#line 547
double __cdecl _yn(int _X, double _Y); 
#line 549
float __cdecl acoshf(float _X); 
#line 550
float __cdecl asinhf(float _X); 
#line 551
float __cdecl atanhf(float _X); 
#line 552
float __cdecl cbrtf(float _X); 
#line 553
float __cdecl _chgsignf(float _X); 
#line 554
float __cdecl copysignf(float _Number, float _Sign); 
#line 555
float __cdecl _copysignf(float _Number, float _Sign); 
#line 556
float __cdecl erff(float _X); 
#line 557
float __cdecl erfcf(float _X); 
#line 558
float __cdecl expm1f(float _X); 
#line 559
float __cdecl exp2f(float _X); 
#line 560
float __cdecl fdimf(float _X, float _Y); 
#line 561
float __cdecl fmaf(float _X, float _Y, float _Z); 
#line 562
float __cdecl fmaxf(float _X, float _Y); 
#line 563
float __cdecl fminf(float _X, float _Y); 
#line 564
float __cdecl _hypotf(float _X, float _Y); 
#line 565
int __cdecl ilogbf(float _X); 
#line 566
float __cdecl lgammaf(float _X); 
#line 567
__int64 __cdecl llrintf(float _X); 
#line 568
__int64 __cdecl llroundf(float _X); 
#line 569
float __cdecl log1pf(float _X); 
#line 570
float __cdecl log2f(float _X); 
#line 571
float __cdecl logbf(float _X); 
#line 572
long __cdecl lrintf(float _X); 
#line 573
long __cdecl lroundf(float _X); 
#line 574
float __cdecl nanf(const char * _X); 
#line 575
float __cdecl nearbyintf(float _X); 
#line 576
float __cdecl nextafterf(float _X, float _Y); 
#line 577
float __cdecl nexttowardf(float _X, long double _Y); 
#line 578
float __cdecl remainderf(float _X, float _Y); 
#line 579
float __cdecl remquof(float _X, float _Y, int * _Z); 
#line 580
float __cdecl rintf(float _X); 
#line 581
float __cdecl roundf(float _X); 
#line 582
float __cdecl scalblnf(float _X, long _Y); 
#line 583
float __cdecl scalbnf(float _X, int _Y); 
#line 584
float __cdecl tgammaf(float _X); 
#line 585
float __cdecl truncf(float _X); 
#line 595 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
float __cdecl _logbf(float _X); 
#line 596
float __cdecl _nextafterf(float _X, float _Y); 
#line 597
int __cdecl _finitef(float _X); 
#line 598
int __cdecl _isnanf(float _X); 
#line 599
int __cdecl _fpclassf(float _X); 
#line 601
int __cdecl _set_FMA3_enable(int _Flag); 
#line 602
int __cdecl _get_FMA3_enable(); 
#line 615 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
float __cdecl acosf(float _X); 
#line 616
float __cdecl asinf(float _X); 
#line 617
float __cdecl atan2f(float _Y, float _X); 
#line 618
float __cdecl atanf(float _X); 
#line 619
float __cdecl ceilf(float _X); 
#line 620
float __cdecl cosf(float _X); 
#line 621
float __cdecl coshf(float _X); 
#line 622
float __cdecl expf(float _X); 
#line 678 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
__inline float __cdecl fabsf(float _X) 
#line 679
{ 
#line 680
return (float)fabs(_X); 
#line 681
} 
#line 687 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
float __cdecl floorf(float _X); 
#line 688
float __cdecl fmodf(float _X, float _Y); 
#line 704 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
__inline float __cdecl frexpf(float _X, int *_Y) 
#line 705
{ 
#line 706
return (float)frexp(_X, _Y); 
#line 707
} 
#line 709
__inline float __cdecl hypotf(float _X, float _Y) 
#line 710
{ 
#line 711
return _hypotf(_X, _Y); 
#line 712
} 
#line 714
__inline float __cdecl ldexpf(float _X, int _Y) 
#line 715
{ 
#line 716
return (float)ldexp(_X, _Y); 
#line 717
} 
#line 721
float __cdecl log10f(float _X); 
#line 722
float __cdecl logf(float _X); 
#line 723
float __cdecl modff(float _X, float * _Y); 
#line 724
float __cdecl powf(float _X, float _Y); 
#line 725
float __cdecl sinf(float _X); 
#line 726
float __cdecl sinhf(float _X); 
#line 727
float __cdecl sqrtf(float _X); 
#line 728
float __cdecl tanf(float _X); 
#line 729
float __cdecl tanhf(float _X); 
#line 783 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
long double __cdecl acoshl(long double _X); 
#line 785
__inline long double __cdecl acosl(long double _X) 
#line 786
{ 
#line 787
return acos((double)_X); 
#line 788
} 
#line 790
long double __cdecl asinhl(long double _X); 
#line 792
__inline long double __cdecl asinl(long double _X) 
#line 793
{ 
#line 794
return asin((double)_X); 
#line 795
} 
#line 797
__inline long double __cdecl atan2l(long double _Y, long double _X) 
#line 798
{ 
#line 799
return atan2((double)_Y, (double)_X); 
#line 800
} 
#line 802
long double __cdecl atanhl(long double _X); 
#line 804
__inline long double __cdecl atanl(long double _X) 
#line 805
{ 
#line 806
return atan((double)_X); 
#line 807
} 
#line 809
long double __cdecl cbrtl(long double _X); 
#line 811
__inline long double __cdecl ceill(long double _X) 
#line 812
{ 
#line 813
return ceil((double)_X); 
#line 814
} 
#line 816
__inline long double __cdecl _chgsignl(long double _X) 
#line 817
{ 
#line 818
return _chgsign((double)_X); 
#line 819
} 
#line 821
long double __cdecl copysignl(long double _Number, long double _Sign); 
#line 823
__inline long double __cdecl _copysignl(long double _Number, long double _Sign) 
#line 824
{ 
#line 825
return _copysign((double)_Number, (double)_Sign); 
#line 826
} 
#line 828
__inline long double __cdecl coshl(long double _X) 
#line 829
{ 
#line 830
return cosh((double)_X); 
#line 831
} 
#line 833
__inline long double __cdecl cosl(long double _X) 
#line 834
{ 
#line 835
return cos((double)_X); 
#line 836
} 
#line 838
long double __cdecl erfl(long double _X); 
#line 839
long double __cdecl erfcl(long double _X); 
#line 841
__inline long double __cdecl expl(long double _X) 
#line 842
{ 
#line 843
return exp((double)_X); 
#line 844
} 
#line 846
long double __cdecl exp2l(long double _X); 
#line 847
long double __cdecl expm1l(long double _X); 
#line 849
__inline long double __cdecl fabsl(long double _X) 
#line 850
{ 
#line 851
return fabs((double)_X); 
#line 852
} 
#line 854
long double __cdecl fdiml(long double _X, long double _Y); 
#line 856
__inline long double __cdecl floorl(long double _X) 
#line 857
{ 
#line 858
return floor((double)_X); 
#line 859
} 
#line 861
long double __cdecl fmal(long double _X, long double _Y, long double _Z); 
#line 862
long double __cdecl fmaxl(long double _X, long double _Y); 
#line 863
long double __cdecl fminl(long double _X, long double _Y); 
#line 865
__inline long double __cdecl fmodl(long double _X, long double _Y) 
#line 866
{ 
#line 867
return fmod((double)_X, (double)_Y); 
#line 868
} 
#line 870
__inline long double __cdecl frexpl(long double _X, int *_Y) 
#line 871
{ 
#line 872
return frexp((double)_X, _Y); 
#line 873
} 
#line 875
int __cdecl ilogbl(long double _X); 
#line 877
__inline long double __cdecl _hypotl(long double _X, long double _Y) 
#line 878
{ 
#line 879
return _hypot((double)_X, (double)_Y); 
#line 880
} 
#line 882
__inline long double __cdecl hypotl(long double _X, long double _Y) 
#line 883
{ 
#line 884
return _hypot((double)_X, (double)_Y); 
#line 885
} 
#line 887
__inline long double __cdecl ldexpl(long double _X, int _Y) 
#line 888
{ 
#line 889
return ldexp((double)_X, _Y); 
#line 890
} 
#line 892
long double __cdecl lgammal(long double _X); 
#line 893
__int64 __cdecl llrintl(long double _X); 
#line 894
__int64 __cdecl llroundl(long double _X); 
#line 896
__inline long double __cdecl logl(long double _X) 
#line 897
{ 
#line 898
return log((double)_X); 
#line 899
} 
#line 901
__inline long double __cdecl log10l(long double _X) 
#line 902
{ 
#line 903
return log10((double)_X); 
#line 904
} 
#line 906
long double __cdecl log1pl(long double _X); 
#line 907
long double __cdecl log2l(long double _X); 
#line 908
long double __cdecl logbl(long double _X); 
#line 909
long __cdecl lrintl(long double _X); 
#line 910
long __cdecl lroundl(long double _X); 
#line 912
__inline long double __cdecl modfl(long double _X, long double *_Y) 
#line 913
{ 
#line 914
double _F, _I; 
#line 915
_F = modf((double)_X, &_I); 
#line 916
(*_Y) = _I; 
#line 917
return _F; 
#line 918
} 
#line 920
long double __cdecl nanl(const char * _X); 
#line 921
long double __cdecl nearbyintl(long double _X); 
#line 922
long double __cdecl nextafterl(long double _X, long double _Y); 
#line 923
long double __cdecl nexttowardl(long double _X, long double _Y); 
#line 925
__inline long double __cdecl powl(long double _X, long double _Y) 
#line 926
{ 
#line 927
return pow((double)_X, (double)_Y); 
#line 928
} 
#line 930
long double __cdecl remainderl(long double _X, long double _Y); 
#line 931
long double __cdecl remquol(long double _X, long double _Y, int * _Z); 
#line 932
long double __cdecl rintl(long double _X); 
#line 933
long double __cdecl roundl(long double _X); 
#line 934
long double __cdecl scalblnl(long double _X, long _Y); 
#line 935
long double __cdecl scalbnl(long double _X, int _Y); 
#line 937
__inline long double __cdecl sinhl(long double _X) 
#line 938
{ 
#line 939
return sinh((double)_X); 
#line 940
} 
#line 942
__inline long double __cdecl sinl(long double _X) 
#line 943
{ 
#line 944
return sin((double)_X); 
#line 945
} 
#line 947
__inline long double __cdecl sqrtl(long double _X) 
#line 948
{ 
#line 949
return sqrt((double)_X); 
#line 950
} 
#line 952
__inline long double __cdecl tanhl(long double _X) 
#line 953
{ 
#line 954
return tanh((double)_X); 
#line 955
} 
#line 957
__inline long double __cdecl tanl(long double _X) 
#line 958
{ 
#line 959
return tan((double)_X); 
#line 960
} 
#line 962
long double __cdecl tgammal(long double _X); 
#line 963
long double __cdecl truncl(long double _X); 
#line 984 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
extern double HUGE; 
#line 989 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
double __cdecl j0(double _X); 
#line 990
double __cdecl j1(double _X); 
#line 991
double __cdecl jn(int _X, double _Y); 
#line 992
double __cdecl y0(double _X); 
#line 993
double __cdecl y1(double _X); 
#line 994
double __cdecl yn(int _X, double _Y); 
#line 999 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\corecrt_math.h"
}__pragma( pack ( pop )) 
#line 1001
#pragma warning(pop)
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_new.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#pragma warning(disable: 4985)
#line 17
extern "C++" {
#line 19
#pragma pack ( push, 8 )
#line 33
namespace std { 
#line 35
struct nothrow_t { 
#line 36
explicit nothrow_t() = default;
#line 37
}; 
#line 39
extern const nothrow_t nothrow; 
#line 40
}
#line 43 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_new.h"
[[nodiscard]] __declspec(allocator) void *__cdecl 
#line 44
operator new(size_t _Size); 
#line 48
[[nodiscard]] __declspec(allocator) void *__cdecl 
#line 49
operator new(size_t _Size, const std::nothrow_t &) noexcept; 
#line 54
[[nodiscard]] __declspec(allocator) void *__cdecl 
#line 55
operator new[](size_t _Size); 
#line 59
[[nodiscard]] __declspec(allocator) void *__cdecl 
#line 60
operator new[](size_t _Size, const std::nothrow_t &) noexcept; 
#line 65
void __cdecl operator delete(void * _Block) noexcept; 
#line 69
void __cdecl operator delete(void * _Block, const std::nothrow_t &) noexcept; 
#line 74
void __cdecl operator delete[](void * _Block) noexcept; 
#line 78
void __cdecl operator delete[](void * _Block, const std::nothrow_t &) noexcept; 
#line 83
void __cdecl operator delete(void * _Block, size_t _Size) noexcept; 
#line 88
void __cdecl operator delete[](void * _Block, size_t _Size) noexcept; 
#line 156
#pragma warning(push)
#pragma warning(disable: 4577)
#pragma warning(disable: 4514)
#line 161
[[nodiscard]] [[msvc::constexpr]] inline void *__cdecl 
#line 162
operator new(size_t _Size, void *
#line 163
_Where) noexcept 
#line 164
{ 
#line 165
(void)_Size; 
#line 166
return _Where; 
#line 167
} 
#line 169
inline void __cdecl operator delete(void *, void *) noexcept 
#line 170
{ 
#line 172
} 
#line 177 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_new.h"
[[nodiscard]] inline void *__cdecl 
#line 178
operator new[](size_t _Size, void *
#line 179
_Where) noexcept 
#line 180
{ 
#line 181
(void)_Size; 
#line 182
return _Where; 
#line 183
} 
#line 185
inline void __cdecl operator delete[](void *, void *) noexcept 
#line 186
{ 
#line 187
} 
#line 195 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_new.h"
}
#line 189
#pragma warning(pop)
#line 193
#pragma pack ( pop )
#line 198 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_new.h"
#pragma warning(pop)
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_new_debug.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 16
extern "C++" {
#line 18
#pragma pack ( push, 8 )
#line 25
[[nodiscard]] 
#line 26
__declspec(allocator) void *__cdecl operator new(size_t _Size, int _BlockUse, const char * _FileName, int _LineNumber); 
#line 33
[[nodiscard]] 
#line 34
__declspec(allocator) void *__cdecl operator new[](size_t _Size, int _BlockUse, const char * _FileName, int _LineNumber); 
#line 41
void __cdecl operator delete(void * _Block, int _BlockUse, const char * _FileName, int _LineNumber) noexcept; 
#line 48
void __cdecl operator delete[](void * _Block, int _BlockUse, const char * _FileName, int _LineNumber) noexcept; 
#line 61 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_new_debug.h"
}
#line 59
#pragma pack ( pop )
#line 64 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\vcruntime_new_debug.h"
#pragma warning(pop)
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\crtdbg.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 19
__pragma( pack ( push, 8 )) extern "C" {
#line 23
typedef void *_HFILE; 
#line 49
typedef int (__cdecl *_CRT_REPORT_HOOK)(int, char *, int *); 
#line 50
typedef int (__cdecl *_CRT_REPORT_HOOKW)(int, __wchar_t *, int *); 
#line 56
typedef int (__cdecl *_CRT_ALLOC_HOOK)(int, void *, size_t, int, long, const unsigned char *, int); 
#line 112
typedef void (__cdecl *_CRT_DUMP_CLIENT)(void *, size_t); 
#line 118
struct _CrtMemBlockHeader; 
#line 127
typedef 
#line 120
struct _CrtMemState { 
#line 122
_CrtMemBlockHeader *pBlockHeader; 
#line 123
size_t lCounts[5]; 
#line 124
size_t lSizes[5]; 
#line 125
size_t lHighWaterCount; 
#line 126
size_t lTotalCount; 
#line 127
} _CrtMemState; 
#line 813 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.22621.0\\ucrt\\crtdbg.h"
}__pragma( pack ( pop )) 
#line 815
#pragma warning(pop)
#line 23 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\yvals.h"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 141 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\yvals.h"
#pragma detect_mismatch("_MSC_VER", "1900")
#line 145 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\yvals.h"
#pragma detect_mismatch("_ITERATOR_DEBUG_LEVEL", "0")
#line 150 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\yvals.h"
#pragma detect_mismatch("RuntimeLibrary", "MT_StaticRelease")
#line 46 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\use_ansi.h"
#pragma comment(lib, "libcpmt")
#line 318 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\yvals.h"
namespace std { 
#line 319
enum _Uninitialized { 
#line 320
_Noinit
#line 321
}; 
#line 323
class _Lockit { 
#line 338
public: __thiscall _Lockit() noexcept; 
#line 339
explicit __thiscall _Lockit(int) noexcept; 
#line 340
__thiscall ~_Lockit() noexcept; 
#line 343 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\yvals.h"
static void __cdecl _Lockit_ctor(int) noexcept; 
#line 344
static void __cdecl _Lockit_dtor(int) noexcept; 
#line 347
private: static void __cdecl _Lockit_ctor(_Lockit *) noexcept; 
#line 348
static void __cdecl _Lockit_ctor(_Lockit *, int) noexcept; 
#line 349
static void __cdecl _Lockit_dtor(_Lockit *) noexcept; 
#line 352
public: _Lockit(const _Lockit &) = delete;
#line 353
_Lockit &operator=(const _Lockit &) = delete;
#line 356
private: int _Locktype; 
#line 357
}; 
#line 459 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\yvals.h"
}
#line 463
#pragma warning(pop)
#pragma pack ( pop )
#line 14 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cstdlib"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 23
[[nodiscard]] inline double abs(double _Xx) noexcept { 
#line 24
return ::fabs(_Xx); 
#line 25
} 
#line 27
[[nodiscard]] inline float abs(float _Xx) noexcept { 
#line 28
return ::fabsf(_Xx); 
#line 29
} 
#line 31
[[nodiscard]] inline long double abs(long double _Xx) noexcept { 
#line 32
return ::fabsl(_Xx); 
#line 33
} 
#line 36
namespace std { 
#line 37
using ::size_t;
#line 38
using ::div_t;
#line 39
using ::ldiv_t;
#line 40
using ::abort;
#line 41
using ::abs;
#line 42
using ::atexit;
#line 43
using ::atof;
#line 44
using ::atoi;
#line 45
using ::atol;
#line 46
using ::bsearch;
#line 47
using ::calloc;
#line 48
using ::div;
#line 49
using ::exit;
#line 50
using ::free;
#line 51
using ::labs;
#line 52
using ::ldiv;
#line 53
using ::malloc;
#line 54
using ::mblen;
#line 55
using ::mbstowcs;
#line 56
using ::mbtowc;
#line 57
using ::qsort;
#line 58
using ::rand;
#line 59
using ::realloc;
#line 60
using ::srand;
#line 61
using ::strtod;
#line 62
using ::strtol;
#line 63
using ::strtoul;
#line 64
using ::wcstombs;
#line 65
using ::wctomb;
#line 67
using ::lldiv_t;
#line 69
using ::getenv;
#line 70
using ::system;
#line 72
using ::atoll;
#line 73
using ::llabs;
#line 74
using ::lldiv;
#line 75
using ::strtof;
#line 76
using ::strtold;
#line 77
using ::strtoll;
#line 78
using ::strtoull;
#line 80
using ::_Exit;
#line 81
using ::at_quick_exit;
#line 82
using ::quick_exit;
#line 83
}
#line 87
#pragma warning(pop)
#pragma pack ( pop )
#line 11 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\xtr1common"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 24
namespace std { 
#line 25
template < class _Ty, _Ty _Val >
struct integral_constant {
    static constexpr _Ty value = _Val;

    using value_type = _Ty;
    using type = integral_constant;

    constexpr operator value_type ( ) const noexcept {
        return value;
    }

    [ [ nodiscard ] ] constexpr value_type operator ( ) ( ) const noexcept {
        return value;
    }
};
#line 41
template< bool _Val> using bool_constant = integral_constant< bool, _Val> ; 
#line 44
using true_type = bool_constant< true> ; 
#line 45
using false_type = bool_constant< false> ; 
#line 47
template< bool _Test, class _Ty = void> 
#line 48
struct enable_if { }; 
#line 50
template< class _Ty> 
#line 51
struct enable_if< true, _Ty>  { 
#line 52
using type = _Ty; 
#line 53
}; 
#line 55
template< bool _Test, class _Ty = void> using enable_if_t = typename enable_if< _Test, _Ty> ::type; 
#line 58
template< bool _Test, class _Ty1, class _Ty2> 
#line 59
struct conditional { 
#line 60
using type = _Ty1; 
#line 61
}; 
#line 63
template< class _Ty1, class _Ty2> 
#line 64
struct conditional< false, _Ty1, _Ty2>  { 
#line 65
using type = _Ty2; 
#line 66
}; 
#line 68
template< bool _Test, class _Ty1, class _Ty2> using conditional_t = typename conditional< _Test, _Ty1, _Ty2> ::type; 
#line 78
template< class , class > constexpr bool 
#line 79
is_same_v = false; 
#line 80
template< class _Ty> constexpr bool 
#line 81
is_same_v< _Ty, _Ty>  = true; 
#line 83
template< class _Ty1, class _Ty2> 
#line 84
struct is_same : public bool_constant< is_same_v< _Ty1, _Ty2> >  { }; 
#line 87 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\xtr1common"
template< class _Ty> 
#line 88
struct remove_const { 
#line 89
using type = _Ty; 
#line 90
}; 
#line 92
template< class _Ty> 
#line 93
struct remove_const< const _Ty>  { 
#line 94
using type = _Ty; 
#line 95
}; 
#line 97
template< class _Ty> using remove_const_t = typename remove_const< _Ty> ::type; 
#line 100
template< class _Ty> 
#line 101
struct remove_volatile { 
#line 102
using type = _Ty; 
#line 103
}; 
#line 105
template< class _Ty> 
#line 106
struct remove_volatile< volatile _Ty>  { 
#line 107
using type = _Ty; 
#line 108
}; 
#line 110
template< class _Ty> using remove_volatile_t = typename remove_volatile< _Ty> ::type; 
#line 113
template< class _Ty> 
#line 114
struct remove_cv { 
#line 115
using type = _Ty; 
#line 117
template< template< class >  class _Fn> using _Apply = _Fn< _Ty> ; 
#line 119
}; 
#line 121
template< class _Ty> 
#line 122
struct remove_cv< const _Ty>  { 
#line 123
using type = _Ty; 
#line 125
template< template< class >  class _Fn> using _Apply = const _Fn< _Ty> ; 
#line 127
}; 
#line 129
template< class _Ty> 
#line 130
struct remove_cv< volatile _Ty>  { 
#line 131
using type = _Ty; 
#line 133
template< template< class >  class _Fn> using _Apply = volatile _Fn< _Ty> ; 
#line 135
}; 
#line 137
template< class _Ty> 
#line 138
struct remove_cv< const volatile _Ty>  { 
#line 139
using type = _Ty; 
#line 141
template< template< class >  class _Fn> using _Apply = const volatile _Fn< _Ty> ; 
#line 143
}; 
#line 145
template< class _Ty> using remove_cv_t = typename remove_cv< _Ty> ::type; 
#line 148
template< bool _First_value, class _First, class ..._Rest> 
#line 149
struct _Disjunction { 
#line 150
using type = _First; 
#line 151
}; 
#line 153
template< class _False, class _Next, class ..._Rest> 
#line 154
struct _Disjunction< false, _False, _Next, _Rest...>  { 
#line 155
using type = typename std::_Disjunction< static_cast< bool>(_Next::value), _Next, _Rest...> ::type; 
#line 156
}; 
#line 158
template< class ..._Traits> 
#line 159
struct disjunction : public false_type { }; 
#line 161
template< class _First, class ..._Rest> 
#line 162
struct disjunction< _First, _Rest...>  : public _Disjunction< static_cast< bool>(_First::value), _First, _Rest...> ::type { 
#line 164
}; 
#line 166
template< class ..._Traits> constexpr bool 
#line 167
disjunction_v = (disjunction< _Traits...> ::value); 
#line 169
template< class _Ty, class ..._Types> constexpr bool 
#line 170
_Is_any_of_v = disjunction_v< is_same< _Ty, _Types> ...> ; 
#line 177 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\xtr1common"
[[nodiscard]] constexpr bool _Is_constant_evaluated() noexcept { 
#line 178
return __builtin_is_constant_evaluated(); 
#line 179
} 
#line 187 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\xtr1common"
template< class _Ty> constexpr bool 
#line 188
is_integral_v = _Is_any_of_v< remove_cv_t< _Ty> , bool, char, signed char, unsigned char, __wchar_t, char16_t, char32_t, short, unsigned short, int, unsigned, long, unsigned long, __int64, unsigned __int64> ; 
#line 194
template< class _Ty> 
#line 195
struct is_integral : public bool_constant< is_integral_v< _Ty> >  { }; 
#line 197
template< class _Ty> constexpr bool 
#line 198
is_floating_point_v = _Is_any_of_v< remove_cv_t< _Ty> , float, double, long double> ; 
#line 200
template< class _Ty> 
#line 201
struct is_floating_point : public bool_constant< is_floating_point_v< _Ty> >  { }; 
#line 203
template< class _Ty> constexpr bool 
#line 204
is_arithmetic_v = is_integral_v< _Ty>  || is_floating_point_v< _Ty> ; 
#line 207
template< class _Ty> 
#line 208
struct is_arithmetic : public bool_constant< is_arithmetic_v< _Ty> >  { }; 
#line 210
template< class _Ty> 
#line 211
struct remove_reference { 
#line 212
using type = _Ty; 
#line 213
using _Const_thru_ref_type = const _Ty; 
#line 214
}; 
#line 216
template< class _Ty> 
#line 217
struct remove_reference< _Ty &>  { 
#line 218
using type = _Ty; 
#line 219
using _Const_thru_ref_type = const _Ty &; 
#line 220
}; 
#line 222
template< class _Ty> 
#line 223
struct remove_reference< _Ty &&>  { 
#line 224
using type = _Ty; 
#line 225
using _Const_thru_ref_type = const _Ty &&; 
#line 226
}; 
#line 228
template< class _Ty> using remove_reference_t = typename remove_reference< _Ty> ::type; 
#line 231
template< class _Ty> using _Const_thru_ref = typename remove_reference< _Ty> ::_Const_thru_ref_type; 
#line 234
template< class _Ty> using _Remove_cvref_t = remove_cv_t< remove_reference_t< _Ty> > ; 
#line 247 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\xtr1common"
}
#line 255
#pragma warning(pop)
#pragma pack ( pop )
#line 29 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 37
[[nodiscard]] inline float acos(float _Xx) noexcept { 
#line 38
return ::acosf(_Xx); 
#line 39
} 
#line 41
[[nodiscard]] inline float acosh(float _Xx) noexcept { 
#line 42
return ::acoshf(_Xx); 
#line 43
} 
#line 45
[[nodiscard]] inline float asin(float _Xx) noexcept { 
#line 46
return ::asinf(_Xx); 
#line 47
} 
#line 49
[[nodiscard]] inline float asinh(float _Xx) noexcept { 
#line 50
return ::asinhf(_Xx); 
#line 51
} 
#line 53
[[nodiscard]] inline float atan(float _Xx) noexcept { 
#line 54
return ::atanf(_Xx); 
#line 55
} 
#line 57
[[nodiscard]] inline float atanh(float _Xx) noexcept { 
#line 58
return ::atanhf(_Xx); 
#line 59
} 
#line 61
[[nodiscard]] inline float atan2(float _Yx, float _Xx) noexcept { 
#line 62
return ::atan2f(_Yx, _Xx); 
#line 63
} 
#line 65
[[nodiscard]] inline float cbrt(float _Xx) noexcept { 
#line 66
return ::cbrtf(_Xx); 
#line 67
} 
#line 69
[[nodiscard]] inline float ceil(float _Xx) noexcept { 
#line 75 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::ceilf(_Xx); 
#line 77 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 79
[[nodiscard]] inline float copysign(float _Number, float _Sign) noexcept { 
#line 85 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::copysignf(_Number, _Sign); 
#line 87 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 89
[[nodiscard]] inline float cos(float _Xx) noexcept { 
#line 90
return ::cosf(_Xx); 
#line 91
} 
#line 93
[[nodiscard]] inline float cosh(float _Xx) noexcept { 
#line 94
return ::coshf(_Xx); 
#line 95
} 
#line 97
[[nodiscard]] inline float erf(float _Xx) noexcept { 
#line 98
return ::erff(_Xx); 
#line 99
} 
#line 101
[[nodiscard]] inline float erfc(float _Xx) noexcept { 
#line 102
return ::erfcf(_Xx); 
#line 103
} 
#line 105
[[nodiscard]] inline float exp(float _Xx) noexcept { 
#line 106
return ::expf(_Xx); 
#line 107
} 
#line 109
[[nodiscard]] inline float exp2(float _Xx) noexcept { 
#line 110
return ::exp2f(_Xx); 
#line 111
} 
#line 113
[[nodiscard]] inline float expm1(float _Xx) noexcept { 
#line 114
return ::expm1f(_Xx); 
#line 115
} 
#line 117
[[nodiscard]] inline float fabs(float _Xx) noexcept { 
#line 118
return ::fabsf(_Xx); 
#line 119
} 
#line 121
[[nodiscard]] inline float fdim(float _Xx, float _Yx) noexcept { 
#line 122
return ::fdimf(_Xx, _Yx); 
#line 123
} 
#line 125
[[nodiscard]] inline float floor(float _Xx) noexcept { 
#line 131 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::floorf(_Xx); 
#line 133 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 135
[[nodiscard]] inline float fma(float _Xx, float _Yx, float _Zx) noexcept { 
#line 136
return ::fmaf(_Xx, _Yx, _Zx); 
#line 137
} 
#line 139
[[nodiscard]] inline float fmax(float _Xx, float _Yx) noexcept { 
#line 140
return ::fmaxf(_Xx, _Yx); 
#line 141
} 
#line 143
[[nodiscard]] inline float fmin(float _Xx, float _Yx) noexcept { 
#line 144
return ::fminf(_Xx, _Yx); 
#line 145
} 
#line 147
[[nodiscard]] inline float fmod(float _Xx, float _Yx) noexcept { 
#line 148
return ::fmodf(_Xx, _Yx); 
#line 149
} 
#line 151
inline float frexp(float _Xx, int *_Yx) noexcept { 
#line 152
return ::frexpf(_Xx, _Yx); 
#line 153
} 
#line 155
[[nodiscard]] inline float hypot(float _Xx, float _Yx) noexcept { 
#line 156
return ::hypotf(_Xx, _Yx); 
#line 157
} 
#line 159
[[nodiscard]] inline int ilogb(float _Xx) noexcept { 
#line 160
return ::ilogbf(_Xx); 
#line 161
} 
#line 163
[[nodiscard]] inline float ldexp(float _Xx, int _Yx) noexcept { 
#line 164
return ::ldexpf(_Xx, _Yx); 
#line 165
} 
#line 167
[[nodiscard]] inline float lgamma(float _Xx) noexcept { 
#line 168
return ::lgammaf(_Xx); 
#line 169
} 
#line 171
[[nodiscard]] inline __int64 llrint(float _Xx) noexcept { 
#line 172
return ::llrintf(_Xx); 
#line 173
} 
#line 175
[[nodiscard]] inline __int64 llround(float _Xx) noexcept { 
#line 176
return ::llroundf(_Xx); 
#line 177
} 
#line 179
[[nodiscard]] inline float log(float _Xx) noexcept { 
#line 180
return ::logf(_Xx); 
#line 181
} 
#line 183
[[nodiscard]] inline float log10(float _Xx) noexcept { 
#line 184
return ::log10f(_Xx); 
#line 185
} 
#line 187
[[nodiscard]] inline float log1p(float _Xx) noexcept { 
#line 188
return ::log1pf(_Xx); 
#line 189
} 
#line 191
[[nodiscard]] inline float log2(float _Xx) noexcept { 
#line 192
return ::log2f(_Xx); 
#line 193
} 
#line 195
[[nodiscard]] inline float logb(float _Xx) noexcept { 
#line 196
return ::logbf(_Xx); 
#line 197
} 
#line 199
[[nodiscard]] inline long lrint(float _Xx) noexcept { 
#line 200
return ::lrintf(_Xx); 
#line 201
} 
#line 203
[[nodiscard]] inline long lround(float _Xx) noexcept { 
#line 204
return ::lroundf(_Xx); 
#line 205
} 
#line 207
inline float modf(float _Xx, float *_Yx) noexcept { 
#line 208
return ::modff(_Xx, _Yx); 
#line 209
} 
#line 211
[[nodiscard]] inline float nearbyint(float _Xx) noexcept { 
#line 212
return ::nearbyintf(_Xx); 
#line 213
} 
#line 215
[[nodiscard]] inline float nextafter(float _Xx, float _Yx) noexcept { 
#line 216
return ::nextafterf(_Xx, _Yx); 
#line 217
} 
#line 219
[[nodiscard]] inline float nexttoward(float _Xx, long double _Yx) noexcept { 
#line 220
return ::nexttowardf(_Xx, _Yx); 
#line 221
} 
#line 223
[[nodiscard]] inline float pow(float _Xx, float _Yx) noexcept { 
#line 224
return ::powf(_Xx, _Yx); 
#line 225
} 
#line 227
[[nodiscard]] inline float remainder(float _Xx, float _Yx) noexcept { 
#line 228
return ::remainderf(_Xx, _Yx); 
#line 229
} 
#line 231
inline float remquo(float _Xx, float _Yx, int *_Zx) noexcept { 
#line 232
return ::remquof(_Xx, _Yx, _Zx); 
#line 233
} 
#line 235
[[nodiscard]] inline float rint(float _Xx) noexcept { 
#line 236
return ::rintf(_Xx); 
#line 237
} 
#line 239
[[nodiscard]] inline float round(float _Xx) noexcept { 
#line 245 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::roundf(_Xx); 
#line 247 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 249
[[nodiscard]] inline float scalbln(float _Xx, long _Yx) noexcept { 
#line 250
return ::scalblnf(_Xx, _Yx); 
#line 251
} 
#line 253
[[nodiscard]] inline float scalbn(float _Xx, int _Yx) noexcept { 
#line 254
return ::scalbnf(_Xx, _Yx); 
#line 255
} 
#line 257
[[nodiscard]] inline float sin(float _Xx) noexcept { 
#line 258
return ::sinf(_Xx); 
#line 259
} 
#line 261
[[nodiscard]] inline float sinh(float _Xx) noexcept { 
#line 262
return ::sinhf(_Xx); 
#line 263
} 
#line 265
[[nodiscard]] inline float sqrt(float _Xx) noexcept { 
#line 266
return ::sqrtf(_Xx); 
#line 267
} 
#line 269
[[nodiscard]] inline float tan(float _Xx) noexcept { 
#line 270
return ::tanf(_Xx); 
#line 271
} 
#line 273
[[nodiscard]] inline float tanh(float _Xx) noexcept { 
#line 274
return ::tanhf(_Xx); 
#line 275
} 
#line 277
[[nodiscard]] inline float tgamma(float _Xx) noexcept { 
#line 278
return ::tgammaf(_Xx); 
#line 279
} 
#line 281
[[nodiscard]] inline float trunc(float _Xx) noexcept { 
#line 287 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::truncf(_Xx); 
#line 289 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 291
[[nodiscard]] inline long double acos(long double _Xx) noexcept { 
#line 292
return ::acosl(_Xx); 
#line 293
} 
#line 295
[[nodiscard]] inline long double acosh(long double _Xx) noexcept { 
#line 296
return ::acoshl(_Xx); 
#line 297
} 
#line 299
[[nodiscard]] inline long double asin(long double _Xx) noexcept { 
#line 300
return ::asinl(_Xx); 
#line 301
} 
#line 303
[[nodiscard]] inline long double asinh(long double _Xx) noexcept { 
#line 304
return ::asinhl(_Xx); 
#line 305
} 
#line 307
[[nodiscard]] inline long double atan(long double _Xx) noexcept { 
#line 308
return ::atanl(_Xx); 
#line 309
} 
#line 311
[[nodiscard]] inline long double atanh(long double _Xx) noexcept { 
#line 312
return ::atanhl(_Xx); 
#line 313
} 
#line 315
[[nodiscard]] inline long double atan2(long double _Yx, long double _Xx) noexcept 
#line 316
{ 
#line 317
return ::atan2l(_Yx, _Xx); 
#line 318
} 
#line 320
[[nodiscard]] inline long double cbrt(long double _Xx) noexcept { 
#line 321
return ::cbrtl(_Xx); 
#line 322
} 
#line 324
[[nodiscard]] inline long double ceil(long double _Xx) noexcept { 
#line 330 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::ceill(_Xx); 
#line 332 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 334
[[nodiscard]] inline long double copysign(long double _Number, long double _Sign) noexcept 
#line 335
{ 
#line 341 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::copysignl(_Number, _Sign); 
#line 343 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 345
[[nodiscard]] inline long double cos(long double _Xx) noexcept { 
#line 346
return ::cosl(_Xx); 
#line 347
} 
#line 349
[[nodiscard]] inline long double cosh(long double _Xx) noexcept { 
#line 350
return ::coshl(_Xx); 
#line 351
} 
#line 353
[[nodiscard]] inline long double erf(long double _Xx) noexcept { 
#line 354
return ::erfl(_Xx); 
#line 355
} 
#line 357
[[nodiscard]] inline long double erfc(long double _Xx) noexcept { 
#line 358
return ::erfcl(_Xx); 
#line 359
} 
#line 361
[[nodiscard]] inline long double exp(long double _Xx) noexcept { 
#line 362
return ::expl(_Xx); 
#line 363
} 
#line 365
[[nodiscard]] inline long double exp2(long double _Xx) noexcept { 
#line 366
return ::exp2l(_Xx); 
#line 367
} 
#line 369
[[nodiscard]] inline long double expm1(long double _Xx) noexcept { 
#line 370
return ::expm1l(_Xx); 
#line 371
} 
#line 373
[[nodiscard]] inline long double fabs(long double _Xx) noexcept { 
#line 374
return ::fabsl(_Xx); 
#line 375
} 
#line 377
[[nodiscard]] inline long double fdim(long double _Xx, long double _Yx) noexcept 
#line 378
{ 
#line 379
return ::fdiml(_Xx, _Yx); 
#line 380
} 
#line 382
[[nodiscard]] inline long double floor(long double _Xx) noexcept { 
#line 388 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::floorl(_Xx); 
#line 390 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 392
[[nodiscard]] inline long double fma(long double 
#line 393
_Xx, long double _Yx, long double _Zx) noexcept { 
#line 394
return ::fmal(_Xx, _Yx, _Zx); 
#line 395
} 
#line 397
[[nodiscard]] inline long double fmax(long double _Xx, long double _Yx) noexcept 
#line 398
{ 
#line 399
return ::fmaxl(_Xx, _Yx); 
#line 400
} 
#line 402
[[nodiscard]] inline long double fmin(long double _Xx, long double _Yx) noexcept 
#line 403
{ 
#line 404
return ::fminl(_Xx, _Yx); 
#line 405
} 
#line 407
[[nodiscard]] inline long double fmod(long double _Xx, long double _Yx) noexcept 
#line 408
{ 
#line 409
return ::fmodl(_Xx, _Yx); 
#line 410
} 
#line 412
inline long double frexp(long double _Xx, int *_Yx) noexcept { 
#line 413
return ::frexpl(_Xx, _Yx); 
#line 414
} 
#line 416
[[nodiscard]] inline long double hypot(long double _Xx, long double _Yx) noexcept 
#line 417
{ 
#line 418
return ::hypotl(_Xx, _Yx); 
#line 419
} 
#line 421
[[nodiscard]] inline int ilogb(long double _Xx) noexcept { 
#line 422
return ::ilogbl(_Xx); 
#line 423
} 
#line 425
[[nodiscard]] inline long double ldexp(long double _Xx, int _Yx) noexcept { 
#line 426
return ::ldexpl(_Xx, _Yx); 
#line 427
} 
#line 429
[[nodiscard]] inline long double lgamma(long double _Xx) noexcept { 
#line 430
return ::lgammal(_Xx); 
#line 431
} 
#line 433
[[nodiscard]] inline __int64 llrint(long double _Xx) noexcept { 
#line 434
return ::llrintl(_Xx); 
#line 435
} 
#line 437
[[nodiscard]] inline __int64 llround(long double _Xx) noexcept { 
#line 438
return ::llroundl(_Xx); 
#line 439
} 
#line 441
[[nodiscard]] inline long double log(long double _Xx) noexcept { 
#line 442
return ::logl(_Xx); 
#line 443
} 
#line 445
[[nodiscard]] inline long double log10(long double _Xx) noexcept { 
#line 446
return ::log10l(_Xx); 
#line 447
} 
#line 449
[[nodiscard]] inline long double log1p(long double _Xx) noexcept { 
#line 450
return ::log1pl(_Xx); 
#line 451
} 
#line 453
[[nodiscard]] inline long double log2(long double _Xx) noexcept { 
#line 454
return ::log2l(_Xx); 
#line 455
} 
#line 457
[[nodiscard]] inline long double logb(long double _Xx) noexcept { 
#line 458
return ::logbl(_Xx); 
#line 459
} 
#line 461
[[nodiscard]] inline long lrint(long double _Xx) noexcept { 
#line 462
return ::lrintl(_Xx); 
#line 463
} 
#line 465
[[nodiscard]] inline long lround(long double _Xx) noexcept { 
#line 466
return ::lroundl(_Xx); 
#line 467
} 
#line 469
inline long double modf(long double _Xx, long double *_Yx) noexcept { 
#line 470
return ::modfl(_Xx, _Yx); 
#line 471
} 
#line 473
[[nodiscard]] inline long double nearbyint(long double _Xx) noexcept { 
#line 474
return ::nearbyintl(_Xx); 
#line 475
} 
#line 477
[[nodiscard]] inline long double nextafter(long double _Xx, long double _Yx) noexcept 
#line 478
{ 
#line 479
return ::nextafterl(_Xx, _Yx); 
#line 480
} 
#line 482
[[nodiscard]] inline long double nexttoward(long double _Xx, long double _Yx) noexcept 
#line 483
{ 
#line 484
return ::nexttowardl(_Xx, _Yx); 
#line 485
} 
#line 487
[[nodiscard]] inline long double pow(long double _Xx, long double _Yx) noexcept 
#line 488
{ 
#line 489
return ::powl(_Xx, _Yx); 
#line 490
} 
#line 492
[[nodiscard]] inline long double remainder(long double _Xx, long double _Yx) noexcept 
#line 493
{ 
#line 494
return ::remainderl(_Xx, _Yx); 
#line 495
} 
#line 497
inline long double remquo(long double _Xx, long double _Yx, int *_Zx) noexcept { 
#line 498
return ::remquol(_Xx, _Yx, _Zx); 
#line 499
} 
#line 501
[[nodiscard]] inline long double rint(long double _Xx) noexcept { 
#line 502
return ::rintl(_Xx); 
#line 503
} 
#line 505
[[nodiscard]] inline long double round(long double _Xx) noexcept { 
#line 511 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::roundl(_Xx); 
#line 513 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 515
[[nodiscard]] inline long double scalbln(long double _Xx, long _Yx) noexcept { 
#line 516
return ::scalblnl(_Xx, _Yx); 
#line 517
} 
#line 519
[[nodiscard]] inline long double scalbn(long double _Xx, int _Yx) noexcept { 
#line 520
return ::scalbnl(_Xx, _Yx); 
#line 521
} 
#line 523
[[nodiscard]] inline long double sin(long double _Xx) noexcept { 
#line 524
return ::sinl(_Xx); 
#line 525
} 
#line 527
[[nodiscard]] inline long double sinh(long double _Xx) noexcept { 
#line 528
return ::sinhl(_Xx); 
#line 529
} 
#line 531
[[nodiscard]] inline long double sqrt(long double _Xx) noexcept { 
#line 532
return ::sqrtl(_Xx); 
#line 533
} 
#line 535
[[nodiscard]] inline long double tan(long double _Xx) noexcept { 
#line 536
return ::tanl(_Xx); 
#line 537
} 
#line 539
[[nodiscard]] inline long double tanh(long double _Xx) noexcept { 
#line 540
return ::tanhl(_Xx); 
#line 541
} 
#line 543
[[nodiscard]] inline long double tgamma(long double _Xx) noexcept { 
#line 544
return ::tgammal(_Xx); 
#line 545
} 
#line 547
[[nodiscard]] inline long double trunc(long double _Xx) noexcept { 
#line 553 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
return ::truncl(_Xx); 
#line 555 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
} 
#line 558
namespace std { 
#line 559
template< class _Ty1, class _Ty2> using _Common_float_type_t = conditional_t< is_same_v< _Ty1, long double>  || is_same_v< _Ty2, long double> , long double, conditional_t< is_same_v< _Ty1, float>  && is_same_v< _Ty2, float> , float, double> > ; 
#line 563
}
#line 566
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 >
double frexp ( _Ty _Value, int * const _Exp ) noexcept {
    return :: frexp ( static_cast < double > ( _Value ), _Exp );
}
#line 571
template < class _Ty1, class _Ty2, class _Ty3,
    :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > && :: std :: is_arithmetic_v < _Ty3 >, int > = 0 >
[ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, :: std :: _Common_float_type_t < _Ty2, _Ty3 > > fma (
    _Ty1 _Left, _Ty2 _Middle, _Ty3 _Right ) noexcept {
    using _Common = :: std :: _Common_float_type_t < _Ty1, :: std :: _Common_float_type_t < _Ty2, _Ty3 >>;
    if constexpr ( :: std :: is_same_v < _Common, float > ) {
        return :: fmaf ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Middle ), static_cast < _Common > ( _Right ) );
    } else if constexpr ( :: std :: is_same_v < _Common, double > ) {
        return :: fma ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Middle ), static_cast < _Common > ( _Right ) );
    } else {
        return :: fmal ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Middle ), static_cast < _Common > ( _Right ) );
    }
}
#line 585
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 >
:: std :: _Common_float_type_t < _Ty1, _Ty2 > remquo ( _Ty1 _Left, _Ty2 _Right, int * _Pquo ) noexcept {
    using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >;
    if constexpr ( :: std :: is_same_v < _Common, float > ) {
        return :: remquof ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ), _Pquo );
    } else if constexpr ( :: std :: is_same_v < _Common, double > ) {
        return :: remquo ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ), _Pquo );
    } else {
        return :: remquol ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ), _Pquo );
    }
}
#line 708
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 >
[ [ nodiscard ] ] inline int fpclassify ( const _Ty _Ix ) noexcept {
    return _Ix == 0 ? 0 : ( - 1 );
}
#line 714
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 >
[ [ nodiscard ] ] inline bool signbit ( const _Ty _Ix ) noexcept {
    if constexpr ( static_cast < _Ty > ( - 1 ) < _Ty { } ) {
        return _Ix < 0;
    } else {
        return false;
    }
}
#line 729
[[nodiscard]] inline bool isnormal(const signed char _Ix) noexcept { return _Ix != 0; } 
#line 730
[[nodiscard]] inline bool isnormal(const unsigned char _Ix) noexcept { return _Ix != 0; } 
#line 731
[[nodiscard]] inline bool isnormal(const short _Ix) noexcept { return _Ix != 0; } 
#line 732
[[nodiscard]] inline bool isnormal(const unsigned short _Ix) noexcept { return _Ix != 0; } 
#line 733
[[nodiscard]] inline bool isnormal(const int _Ix) noexcept { return _Ix != 0; } 
#line 734
[[nodiscard]] inline bool isnormal(const unsigned _Ix) noexcept { return _Ix != (0); } 
#line 735
[[nodiscard]] inline bool isnormal(const long _Ix) noexcept { return _Ix != (0); } 
#line 736
[[nodiscard]] inline bool isnormal(const unsigned long _Ix) noexcept { return _Ix != (0); } 
#line 737
[[nodiscard]] inline bool isnormal(const __int64 _Ix) noexcept { return _Ix != (0); } 
#line 738
[[nodiscard]] inline bool isnormal(const unsigned __int64 _Ix) noexcept { return _Ix != (0); } 
#line 739
[[nodiscard]] inline bool isnormal(const bool _Ix) noexcept { return _Ix != 0; } 
#line 740
[[nodiscard]] inline bool isnormal(const char _Ix) noexcept { return _Ix != 0; } 
#line 744
[[nodiscard]] inline bool isnormal(const char16_t _Ix) noexcept { return _Ix != 0; } 
#line 745
[[nodiscard]] inline bool isnormal(const char32_t _Ix) noexcept { return _Ix != (0); } 
#line 747
[[nodiscard]] inline bool isnormal(const __wchar_t _Ix) noexcept { return _Ix != 0; } 
#line 834 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double acos ( _Ty _Left ) noexcept { return :: acos ( static_cast < double > ( _Left ) ); }
#line 835
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double asin ( _Ty _Left ) noexcept { return :: asin ( static_cast < double > ( _Left ) ); }
#line 836
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double atan ( _Ty _Left ) noexcept { return :: atan ( static_cast < double > ( _Left ) ); }
#line 837
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > atan2 ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: atan2 ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 838
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double cos ( _Ty _Left ) noexcept { return :: cos ( static_cast < double > ( _Left ) ); }
#line 839
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double sin ( _Ty _Left ) noexcept { return :: sin ( static_cast < double > ( _Left ) ); }
#line 840
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double tan ( _Ty _Left ) noexcept { return :: tan ( static_cast < double > ( _Left ) ); }
#line 841
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double acosh ( _Ty _Left ) noexcept { return :: acosh ( static_cast < double > ( _Left ) ); }
#line 842
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double asinh ( _Ty _Left ) noexcept { return :: asinh ( static_cast < double > ( _Left ) ); }
#line 843
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double atanh ( _Ty _Left ) noexcept { return :: atanh ( static_cast < double > ( _Left ) ); }
#line 844
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double cosh ( _Ty _Left ) noexcept { return :: cosh ( static_cast < double > ( _Left ) ); }
#line 845
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double sinh ( _Ty _Left ) noexcept { return :: sinh ( static_cast < double > ( _Left ) ); }
#line 846
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double tanh ( _Ty _Left ) noexcept { return :: tanh ( static_cast < double > ( _Left ) ); }
#line 847
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double exp ( _Ty _Left ) noexcept { return :: exp ( static_cast < double > ( _Left ) ); }
#line 848
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double exp2 ( _Ty _Left ) noexcept { return :: exp2 ( static_cast < double > ( _Left ) ); }
#line 849
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double expm1 ( _Ty _Left ) noexcept { return :: expm1 ( static_cast < double > ( _Left ) ); }
#line 851
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] int ilogb ( _Ty _Left ) noexcept { return :: ilogb ( static_cast < double > ( _Left ) ); }
#line 852
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double ldexp ( _Ty _Left, int _Arg2 ) noexcept { return :: ldexp ( static_cast < double > ( _Left ), _Arg2 ); }
#line 853
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double log ( _Ty _Left ) noexcept { return :: log ( static_cast < double > ( _Left ) ); }
#line 854
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double log10 ( _Ty _Left ) noexcept { return :: log10 ( static_cast < double > ( _Left ) ); }
#line 855
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double log1p ( _Ty _Left ) noexcept { return :: log1p ( static_cast < double > ( _Left ) ); }
#line 856
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double log2 ( _Ty _Left ) noexcept { return :: log2 ( static_cast < double > ( _Left ) ); }
#line 857
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double logb ( _Ty _Left ) noexcept { return :: logb ( static_cast < double > ( _Left ) ); }
#line 859
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double scalbn ( _Ty _Left, int _Arg2 ) noexcept { return :: scalbn ( static_cast < double > ( _Left ), _Arg2 ); }
#line 860
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double scalbln ( _Ty _Left, long _Arg2 ) noexcept { return :: scalbln ( static_cast < double > ( _Left ), _Arg2 ); }
#line 861
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double cbrt ( _Ty _Left ) noexcept { return :: cbrt ( static_cast < double > ( _Left ) ); }
#line 863
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double fabs ( _Ty _Left ) noexcept { return :: fabs ( static_cast < double > ( _Left ) ); }
#line 864
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > hypot ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: hypot ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 866
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > pow ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: pow ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 867
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double sqrt ( _Ty _Left ) noexcept { return :: sqrt ( static_cast < double > ( _Left ) ); }
#line 868
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double erf ( _Ty _Left ) noexcept { return :: erf ( static_cast < double > ( _Left ) ); }
#line 869
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double erfc ( _Ty _Left ) noexcept { return :: erfc ( static_cast < double > ( _Left ) ); }
#line 870
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double lgamma ( _Ty _Left ) noexcept { return :: lgamma ( static_cast < double > ( _Left ) ); }
#line 871
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double tgamma ( _Ty _Left ) noexcept { return :: tgamma ( static_cast < double > ( _Left ) ); }
#line 872
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double ceil ( _Ty _Left ) noexcept { return :: ceil ( static_cast < double > ( _Left ) ); }
#line 873
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double floor ( _Ty _Left ) noexcept { return :: floor ( static_cast < double > ( _Left ) ); }
#line 874
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double nearbyint ( _Ty _Left ) noexcept { return :: nearbyint ( static_cast < double > ( _Left ) ); }
#line 875
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double rint ( _Ty _Left ) noexcept { return :: rint ( static_cast < double > ( _Left ) ); }
#line 876
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] long lrint ( _Ty _Left ) noexcept { return :: lrint ( static_cast < double > ( _Left ) ); }
#line 877
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] long long llrint ( _Ty _Left ) noexcept { return :: llrint ( static_cast < double > ( _Left ) ); }
#line 878
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double round ( _Ty _Left ) noexcept { return :: round ( static_cast < double > ( _Left ) ); }
#line 879
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] long lround ( _Ty _Left ) noexcept { return :: lround ( static_cast < double > ( _Left ) ); }
#line 880
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] long long llround ( _Ty _Left ) noexcept { return :: llround ( static_cast < double > ( _Left ) ); }
#line 881
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double trunc ( _Ty _Left ) noexcept { return :: trunc ( static_cast < double > ( _Left ) ); }
#line 882
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmod ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: fmod ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 883
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > remainder ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: remainder ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 885
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > copysign ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: copysign ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 887
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > nextafter ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: nextafter ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 888
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double nexttoward ( _Ty _Left, long double _Arg2 ) noexcept { return :: nexttoward ( static_cast < double > ( _Left ), _Arg2 ); }
#line 889
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > fdim ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: fdim ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 890
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmax ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: fmax ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 891
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmin ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: fmin ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 895
[[nodiscard]] inline bool isfinite(signed char) noexcept { return true; } [[nodiscard]] inline bool isfinite(unsigned char) noexcept { return true; } [[nodiscard]] inline bool isfinite(short) noexcept { return true; } [[nodiscard]] inline bool isfinite(unsigned short) noexcept { return true; } [[nodiscard]] inline bool isfinite(int) noexcept { return true; } [[nodiscard]] inline bool isfinite(unsigned) noexcept { return true; } [[nodiscard]] inline bool isfinite(long) noexcept { return true; } [[nodiscard]] inline bool isfinite(unsigned long) noexcept { return true; } [[nodiscard]] inline bool isfinite(__int64) noexcept { return true; } [[nodiscard]] inline bool isfinite(unsigned __int64) noexcept { return true; } [[nodiscard]] inline bool isfinite(bool) noexcept { return true; } [[nodiscard]] inline bool isfinite(char) noexcept { return true; } [[nodiscard]] inline bool isfinite(char16_t) noexcept { return true; } [[nodiscard]] inline bool isfinite(char32_t) noexcept { return true; } [[nodiscard]] inline bool isfinite(__wchar_t) noexcept { return true; } 
#line 896
[[nodiscard]] inline bool isinf(signed char) noexcept { return false; } [[nodiscard]] inline bool isinf(unsigned char) noexcept { return false; } [[nodiscard]] inline bool isinf(short) noexcept { return false; } [[nodiscard]] inline bool isinf(unsigned short) noexcept { return false; } [[nodiscard]] inline bool isinf(int) noexcept { return false; } [[nodiscard]] inline bool isinf(unsigned) noexcept { return false; } [[nodiscard]] inline bool isinf(long) noexcept { return false; } [[nodiscard]] inline bool isinf(unsigned long) noexcept { return false; } [[nodiscard]] inline bool isinf(__int64) noexcept { return false; } [[nodiscard]] inline bool isinf(unsigned __int64) noexcept { return false; } [[nodiscard]] inline bool isinf(bool) noexcept { return false; } [[nodiscard]] inline bool isinf(char) noexcept { return false; } [[nodiscard]] inline bool isinf(char16_t) noexcept { return false; } [[nodiscard]] inline bool isinf(char32_t) noexcept { return false; } [[nodiscard]] inline bool isinf(__wchar_t) noexcept { return false; } 
#line 897
[[nodiscard]] inline bool isnan(signed char) noexcept { return false; } [[nodiscard]] inline bool isnan(unsigned char) noexcept { return false; } [[nodiscard]] inline bool isnan(short) noexcept { return false; } [[nodiscard]] inline bool isnan(unsigned short) noexcept { return false; } [[nodiscard]] inline bool isnan(int) noexcept { return false; } [[nodiscard]] inline bool isnan(unsigned) noexcept { return false; } [[nodiscard]] inline bool isnan(long) noexcept { return false; } [[nodiscard]] inline bool isnan(unsigned long) noexcept { return false; } [[nodiscard]] inline bool isnan(__int64) noexcept { return false; } [[nodiscard]] inline bool isnan(unsigned __int64) noexcept { return false; } [[nodiscard]] inline bool isnan(bool) noexcept { return false; } [[nodiscard]] inline bool isnan(char) noexcept { return false; } [[nodiscard]] inline bool isnan(char16_t) noexcept { return false; } [[nodiscard]] inline bool isnan(char32_t) noexcept { return false; } [[nodiscard]] inline bool isnan(__wchar_t) noexcept { return false; } 
#line 916
namespace std { 
#line 917
using ::abs;
#line 918
using ::acos;
#line 919
using ::asin;
#line 920
using ::atan;
#line 921
using ::atan2;
#line 922
using ::ceil;
#line 923
using ::cos;
#line 924
using ::cosh;
#line 925
using ::exp;
#line 926
using ::fabs;
#line 927
using ::floor;
#line 928
using ::fmod;
#line 929
using ::frexp;
#line 930
using ::ldexp;
#line 931
using ::log;
#line 932
using ::log10;
#line 933
using ::modf;
#line 934
using ::pow;
#line 935
using ::sin;
#line 936
using ::sinh;
#line 937
using ::sqrt;
#line 938
using ::tan;
#line 939
using ::tanh;
#line 941
using ::acosf;
#line 942
using ::asinf;
#line 943
using ::atanf;
#line 944
using ::atan2f;
#line 945
using ::ceilf;
#line 946
using ::cosf;
#line 947
using ::coshf;
#line 948
using ::expf;
#line 949
using ::fabsf;
#line 950
using ::floorf;
#line 951
using ::fmodf;
#line 952
using ::frexpf;
#line 953
using ::ldexpf;
#line 954
using ::logf;
#line 955
using ::log10f;
#line 956
using ::modff;
#line 957
using ::powf;
#line 958
using ::sinf;
#line 959
using ::sinhf;
#line 960
using ::sqrtf;
#line 961
using ::tanf;
#line 962
using ::tanhf;
#line 964
using ::acosl;
#line 965
using ::asinl;
#line 966
using ::atanl;
#line 967
using ::atan2l;
#line 968
using ::ceill;
#line 969
using ::cosl;
#line 970
using ::coshl;
#line 971
using ::expl;
#line 972
using ::fabsl;
#line 973
using ::floorl;
#line 974
using ::fmodl;
#line 975
using ::frexpl;
#line 976
using ::ldexpl;
#line 977
using ::logl;
#line 978
using ::log10l;
#line 979
using ::modfl;
#line 980
using ::powl;
#line 981
using ::sinl;
#line 982
using ::sinhl;
#line 983
using ::sqrtl;
#line 984
using ::tanl;
#line 985
using ::tanhl;
#line 987
using ::float_t;
#line 988
using ::double_t;
#line 990
using ::acosh;
#line 991
using ::asinh;
#line 992
using ::atanh;
#line 993
using ::cbrt;
#line 994
using ::erf;
#line 995
using ::erfc;
#line 996
using ::expm1;
#line 997
using ::exp2;
#line 999
inline namespace _Binary_hypot { 
#line 1000
using ::hypot;
#line 1001
}
#line 1003
using ::ilogb;
#line 1004
using ::lgamma;
#line 1005
using ::log1p;
#line 1006
using ::log2;
#line 1007
using ::logb;
#line 1008
using ::llrint;
#line 1009
using ::lrint;
#line 1010
using ::nearbyint;
#line 1011
using ::rint;
#line 1012
using ::llround;
#line 1013
using ::lround;
#line 1014
using ::fdim;
#line 1015
using ::fma;
#line 1016
using ::fmax;
#line 1017
using ::fmin;
#line 1018
using ::round;
#line 1019
using ::trunc;
#line 1020
using ::remainder;
#line 1021
using ::remquo;
#line 1022
using ::copysign;
#line 1023
using ::nan;
#line 1024
using ::nextafter;
#line 1025
using ::scalbn;
#line 1026
using ::scalbln;
#line 1027
using ::nexttoward;
#line 1028
using ::tgamma;
#line 1030
using ::acoshf;
#line 1031
using ::asinhf;
#line 1032
using ::atanhf;
#line 1033
using ::cbrtf;
#line 1034
using ::erff;
#line 1035
using ::erfcf;
#line 1036
using ::expm1f;
#line 1037
using ::exp2f;
#line 1038
using ::hypotf;
#line 1039
using ::ilogbf;
#line 1040
using ::lgammaf;
#line 1041
using ::log1pf;
#line 1042
using ::log2f;
#line 1043
using ::logbf;
#line 1044
using ::llrintf;
#line 1045
using ::lrintf;
#line 1046
using ::nearbyintf;
#line 1047
using ::rintf;
#line 1048
using ::llroundf;
#line 1049
using ::lroundf;
#line 1050
using ::fdimf;
#line 1051
using ::fmaf;
#line 1052
using ::fmaxf;
#line 1053
using ::fminf;
#line 1054
using ::roundf;
#line 1055
using ::truncf;
#line 1056
using ::remainderf;
#line 1057
using ::remquof;
#line 1058
using ::copysignf;
#line 1059
using ::nanf;
#line 1060
using ::nextafterf;
#line 1061
using ::scalbnf;
#line 1062
using ::scalblnf;
#line 1063
using ::nexttowardf;
#line 1064
using ::tgammaf;
#line 1066
using ::acoshl;
#line 1067
using ::asinhl;
#line 1068
using ::atanhl;
#line 1069
using ::cbrtl;
#line 1070
using ::erfl;
#line 1071
using ::erfcl;
#line 1072
using ::expm1l;
#line 1073
using ::exp2l;
#line 1074
using ::hypotl;
#line 1075
using ::ilogbl;
#line 1076
using ::lgammal;
#line 1077
using ::log1pl;
#line 1078
using ::log2l;
#line 1079
using ::logbl;
#line 1080
using ::llrintl;
#line 1081
using ::lrintl;
#line 1082
using ::nearbyintl;
#line 1083
using ::rintl;
#line 1084
using ::llroundl;
#line 1085
using ::lroundl;
#line 1086
using ::fdiml;
#line 1087
using ::fmal;
#line 1088
using ::fmaxl;
#line 1089
using ::fminl;
#line 1090
using ::roundl;
#line 1091
using ::truncl;
#line 1092
using ::remainderl;
#line 1093
using ::remquol;
#line 1094
using ::copysignl;
#line 1095
using ::nanl;
#line 1096
using ::nextafterl;
#line 1097
using ::scalbnl;
#line 1098
using ::scalblnl;
#line 1099
using ::nexttowardl;
#line 1100
using ::tgammal;
#line 1102
using ::fpclassify;
#line 1103
using ::signbit;
#line 1104
using ::isfinite;
#line 1105
using ::isinf;
#line 1106
using ::isnan;
#line 1107
using ::isnormal;
#line 1108
using ::isgreater;
#line 1109
using ::isgreaterequal;
#line 1110
using ::isless;
#line 1111
using ::islessequal;
#line 1112
using ::islessgreater;
#line 1113
using ::isunordered;
#line 1114
}
#line 1697 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cmath"
#pragma warning(pop)
#pragma pack ( pop )
#line 4866 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern "C" double __cdecl _hypot(double x, double y); 
#line 4867
extern "C" float __cdecl _hypotf(float x, float y); 
#line 4877 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline bool signbit(long double) throw(); 
#line 4878
extern "C" int _ldsign(long double); 
#line 4921 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline bool signbit(double) throw(); 
#line 4922
extern "C" int _dsign(double); 
#line 4966 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline bool signbit(float) throw(); 
#line 4967
extern "C" int _fdsign(float); 
#line 4975 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isinf(long double a); 
#line 5010 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isinf(double a); 
#line 5048 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isinf(float a); 
#line 5055 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isnan(long double a); 
#line 5088 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isnan(double a); 
#line 5124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isnan(float a); 
#line 5131 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isfinite(long double a); 
#line 5168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isfinite(double a); 
#line 5204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static __inline bool isfinite(float a); 
#line 5222 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
template< class T> extern T _Pow_int(T, int) throw(); 
#line 5223
extern inline __int64 abs(__int64) throw(); 
#line 5314 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline long __cdecl abs(long) throw(); 
#line 5318 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline float __cdecl abs(float) throw(); 
#line 5319
extern inline double __cdecl abs(double) throw(); 
#line 5320
extern inline float __cdecl fabs(float) throw(); 
#line 5321
extern inline float __cdecl ceil(float) throw(); 
#line 5322
extern inline float __cdecl floor(float) throw(); 
#line 5323
extern inline float __cdecl sqrt(float) throw(); 
#line 5324
extern inline float __cdecl pow(float, float) throw(); 
#line 5363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
template< class _Ty1, class _Ty2, std::enable_if_t< std::is_arithmetic_v< _Ty1>  && std::is_arithmetic_v< _Ty2> , int>  > [[nodiscard]] std::_Common_float_type_t< _Ty1, _Ty2>  __cdecl pow(_Ty1 _Left, _Ty2 _Right) noexcept; 
#line 5370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline float __cdecl log(float) throw(); 
#line 5371
extern inline float __cdecl log10(float) throw(); 
#line 5372
extern inline float __cdecl fmod(float, float) throw(); 
#line 5373
extern inline float __cdecl modf(float, float *) throw(); 
#line 5374
extern inline float __cdecl exp(float) throw(); 
#line 5375
extern inline float __cdecl frexp(float, int *) throw(); 
#line 5376
extern inline float __cdecl ldexp(float, int) throw(); 
#line 5377
extern inline float __cdecl asin(float) throw(); 
#line 5378
extern inline float __cdecl sin(float) throw(); 
#line 5379
extern inline float __cdecl sinh(float) throw(); 
#line 5380
extern inline float __cdecl acos(float) throw(); 
#line 5381
extern inline float __cdecl cos(float) throw(); 
#line 5382
extern inline float __cdecl cosh(float) throw(); 
#line 5383
extern inline float __cdecl atan(float) throw(); 
#line 5384
extern inline float __cdecl atan2(float, float) throw(); 
#line 5385
extern inline float __cdecl tan(float) throw(); 
#line 5386
extern inline float __cdecl tanh(float) throw(); 
#line 5609 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
extern inline float __cdecl logb(float) throw(); 
#line 5610
extern inline int __cdecl ilogb(float) throw(); 
#line 5611
extern float __cdecl scalbn(float, float) throw(); 
#line 5612
extern inline float __cdecl scalbln(float, long) throw(); 
#line 5613
extern inline float __cdecl exp2(float) throw(); 
#line 5614
extern inline float __cdecl expm1(float) throw(); 
#line 5615
extern inline float __cdecl log2(float) throw(); 
#line 5616
extern inline float __cdecl log1p(float) throw(); 
#line 5617
extern inline float __cdecl acosh(float) throw(); 
#line 5618
extern inline float __cdecl asinh(float) throw(); 
#line 5619
extern inline float __cdecl atanh(float) throw(); 
#line 5620
extern inline float __cdecl hypot(float, float) throw(); 
#line 5621
extern inline float __cdecl cbrt(float) throw(); 
#line 5622
extern inline float __cdecl erf(float) throw(); 
#line 5623
extern inline float __cdecl erfc(float) throw(); 
#line 5624
extern inline float __cdecl lgamma(float) throw(); 
#line 5625
extern inline float __cdecl tgamma(float) throw(); 
#line 5626
extern inline float __cdecl copysign(float, float) throw(); 
#line 5627
extern inline float __cdecl nextafter(float, float) throw(); 
#line 5628
extern inline float __cdecl remainder(float, float) throw(); 
#line 5629
extern inline float __cdecl remquo(float, float, int *) throw(); 
#line 5630
extern inline float __cdecl round(float) throw(); 
#line 5631
extern inline long __cdecl lround(float) throw(); 
#line 5632
extern inline __int64 __cdecl llround(float) throw(); 
#line 5633
extern inline float __cdecl trunc(float) throw(); 
#line 5634
extern inline float __cdecl rint(float) throw(); 
#line 5635
extern inline long __cdecl lrint(float) throw(); 
#line 5636
extern inline __int64 __cdecl llrint(float) throw(); 
#line 5637
extern inline float __cdecl nearbyint(float) throw(); 
#line 5638
extern inline float __cdecl fdim(float, float) throw(); 
#line 5639
extern inline float __cdecl fma(float, float, float) throw(); 
#line 5640
extern inline float __cdecl fmax(float, float) throw(); 
#line 5641
extern inline float __cdecl fmin(float, float) throw(); 
#line 5644 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
static inline float exp10(const float a); 
#line 5646
static inline float rsqrt(const float a); 
#line 5648
static inline float rcbrt(const float a); 
#line 5650
static inline float sinpi(const float a); 
#line 5652
static inline float cospi(const float a); 
#line 5654
static inline void sincospi(const float a, float *const sptr, float *const cptr); 
#line 5656
static inline void sincos(const float a, float *const sptr, float *const cptr); 
#line 5658
static inline float j0(const float a); 
#line 5660
static inline float j1(const float a); 
#line 5662
static inline float jn(const int n, const float a); 
#line 5664
static inline float y0(const float a); 
#line 5666
static inline float y1(const float a); 
#line 5668
static inline float yn(const int n, const float a); 
#line 5670
static inline float cyl_bessel_i0(const float a); 
#line 5672
static inline float cyl_bessel_i1(const float a); 
#line 5674
static inline float erfinv(const float a); 
#line 5676
static inline float erfcinv(const float a); 
#line 5678
static inline float normcdfinv(const float a); 
#line 5680
static inline float normcdf(const float a); 
#line 5682
static inline float erfcx(const float a); 
#line 5684
static inline double copysign(const double a, const float b); 
#line 5686
static inline double copysign(const float a, const double b); 
#line 5694
static inline unsigned min(const unsigned a, const unsigned b); 
#line 5702
static inline unsigned min(const int a, const unsigned b); 
#line 5710
static inline unsigned min(const unsigned a, const int b); 
#line 5718
static inline long min(const long a, const long b); 
#line 5726
static inline unsigned long min(const unsigned long a, const unsigned long b); 
#line 5734
static inline unsigned long min(const long a, const unsigned long b); 
#line 5742
static inline unsigned long min(const unsigned long a, const long b); 
#line 5750
static inline __int64 min(const __int64 a, const __int64 b); 
#line 5758
static inline unsigned __int64 min(const unsigned __int64 a, const unsigned __int64 b); 
#line 5766
static inline unsigned __int64 min(const __int64 a, const unsigned __int64 b); 
#line 5774
static inline unsigned __int64 min(const unsigned __int64 a, const __int64 b); 
#line 5785
static inline float min(const float a, const float b); 
#line 5796
static inline double min(const double a, const double b); 
#line 5806
static inline double min(const float a, const double b); 
#line 5816
static inline double min(const double a, const float b); 
#line 5824
static inline unsigned max(const unsigned a, const unsigned b); 
#line 5832
static inline unsigned max(const int a, const unsigned b); 
#line 5840
static inline unsigned max(const unsigned a, const int b); 
#line 5848
static inline long max(const long a, const long b); 
#line 5856
static inline unsigned long max(const unsigned long a, const unsigned long b); 
#line 5864
static inline unsigned long max(const long a, const unsigned long b); 
#line 5872
static inline unsigned long max(const unsigned long a, const long b); 
#line 5880
static inline __int64 max(const __int64 a, const __int64 b); 
#line 5888
static inline unsigned __int64 max(const unsigned __int64 a, const unsigned __int64 b); 
#line 5896
static inline unsigned __int64 max(const __int64 a, const unsigned __int64 b); 
#line 5904
static inline unsigned __int64 max(const unsigned __int64 a, const __int64 b); 
#line 5915
static inline float max(const float a, const float b); 
#line 5926
static inline double max(const double a, const double b); 
#line 5936
static inline double max(const float a, const double b); 
#line 5946
static inline double max(const double a, const float b); 
#line 5958
extern "C" {
#line 5959
inline void *__nv_aligned_device_malloc(size_t size, size_t align) {int volatile ___ = 1;(void)size;(void)align;::exit(___);}
#if 0
#line 5960
{ 
#line 5961
void *__nv_aligned_device_malloc_impl(size_t, size_t); 
#line 5962
return __nv_aligned_device_malloc_impl(size, align); 
#line 5963
} 
#endif
#line 5964 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.h"
}
#line 433 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const long double a) 
#line 434
{ 
#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isinf< long double> (a); 
#line 440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 449 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const double a) 
#line 450
{ 
#line 454 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isinf< double> (a); 
#line 456 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 465 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const float a) 
#line 466
{ 
#line 470 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isinf< float> (a); 
#line 472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 481 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const long double a) 
#line 482
{ 
#line 486 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isnan< long double> (a); 
#line 488 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 497 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const double a) 
#line 498
{ 
#line 502 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isnan< double> (a); 
#line 504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const float a) 
#line 514
{ 
#line 518 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isnan< float> (a); 
#line 520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const long double a) 
#line 530
{ 
#line 534 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isfinite< long double> (a); 
#line 536 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 545 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const double a) 
#line 546
{ 
#line 550 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isfinite< double> (a); 
#line 552 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 561 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const float a) 
#line 562
{ 
#line 566 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
return isfinite< float> (a); 
#line 568 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
} 
#line 758 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static inline float exp10(const float a) 
#line 759
{ 
#line 760
return exp10f(a); 
#line 761
} 
#line 763
static inline float rsqrt(const float a) 
#line 764
{ 
#line 765
return rsqrtf(a); 
#line 766
} 
#line 768
static inline float rcbrt(const float a) 
#line 769
{ 
#line 770
return rcbrtf(a); 
#line 771
} 
#line 773
static inline float sinpi(const float a) 
#line 774
{ 
#line 775
return sinpif(a); 
#line 776
} 
#line 778
static inline float cospi(const float a) 
#line 779
{ 
#line 780
return cospif(a); 
#line 781
} 
#line 783
static inline void sincospi(const float a, float *const sptr, float *const cptr) 
#line 784
{ 
#line 785
sincospif(a, sptr, cptr); 
#line 786
} 
#line 788
static inline void sincos(const float a, float *const sptr, float *const cptr) 
#line 789
{ 
#line 790
sincosf(a, sptr, cptr); 
#line 791
} 
#line 793
static inline float j0(const float a) 
#line 794
{ 
#line 795
return j0f(a); 
#line 796
} 
#line 798
static inline float j1(const float a) 
#line 799
{ 
#line 800
return j1f(a); 
#line 801
} 
#line 803
static inline float jn(const int n, const float a) 
#line 804
{ 
#line 805
return jnf(n, a); 
#line 806
} 
#line 808
static inline float y0(const float a) 
#line 809
{ 
#line 810
return y0f(a); 
#line 811
} 
#line 813
static inline float y1(const float a) 
#line 814
{ 
#line 815
return y1f(a); 
#line 816
} 
#line 818
static inline float yn(const int n, const float a) 
#line 819
{ 
#line 820
return ynf(n, a); 
#line 821
} 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 823
static inline float cyl_bessel_i0(const float a) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 824
{ 
#line 825
return cyl_bessel_i0f(a); 
#line 826
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 828 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static inline float cyl_bessel_i1(const float a) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 829
{ 
#line 830
return cyl_bessel_i1f(a); 
#line 831
} 
#endif
#line 833 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static inline float erfinv(const float a) 
#line 834
{ 
#line 835
return erfinvf(a); 
#line 836
} 
#line 838
static inline float erfcinv(const float a) 
#line 839
{ 
#line 840
return erfcinvf(a); 
#line 841
} 
#line 843
static inline float normcdfinv(const float a) 
#line 844
{ 
#line 845
return normcdfinvf(a); 
#line 846
} 
#line 848
static inline float normcdf(const float a) 
#line 849
{ 
#line 850
return normcdff(a); 
#line 851
} 
#line 853
static inline float erfcx(const float a) 
#line 854
{ 
#line 855
return erfcxf(a); 
#line 856
} 
#line 858
static inline double copysign(const double a, const float b) 
#line 859
{ 
#line 860
return copysign(a, static_cast< double>(b)); 
#line 861
} 
#line 863
static inline double copysign(const float a, const double b) 
#line 864
{ 
#line 865
return copysign(static_cast< double>(a), b); 
#line 866
} 
#line 868
static inline unsigned min(const unsigned a, const unsigned b) 
#line 869
{ 
#line 870
return umin(a, b); 
#line 871
} 
#line 873
static inline unsigned min(const int a, const unsigned b) 
#line 874
{ 
#line 875
return umin(static_cast< unsigned>(a), b); 
#line 876
} 
#line 878
static inline unsigned min(const unsigned a, const int b) 
#line 879
{ 
#line 880
return umin(a, static_cast< unsigned>(b)); 
#line 881
} 
#line 883
static inline long min(const long a, const long b) 
#line 884
{ 
#line 885
long retval; 
#line 888
#pragma warning (push)
#pragma warning (disable: 4127)
#line 892 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 894
#pragma warning (pop)
#line 896 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< long>(min(static_cast< int>(a), static_cast< int>(b)))); 
#line 897
} else { 
#line 898
retval = (static_cast< long>(llmin(static_cast< __int64>(a), static_cast< __int64>(b)))); 
#line 899
}  
#line 900
return retval; 
#line 901
} 
#line 903
static inline unsigned long min(const unsigned long a, const unsigned long b) 
#line 904
{ 
#line 905
unsigned long retval; 
#line 907
#pragma warning (push)
#pragma warning (disable: 4127)
#line 910 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 912
#pragma warning (pop)
#line 914 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 915
} else { 
#line 916
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 917
}  
#line 918
return retval; 
#line 919
} 
#line 921
static inline unsigned long min(const long a, const unsigned long b) 
#line 922
{ 
#line 923
unsigned long retval; 
#line 925
#pragma warning (push)
#pragma warning (disable: 4127)
#line 928 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 930
#pragma warning (pop)
#line 932 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 933
} else { 
#line 934
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 935
}  
#line 936
return retval; 
#line 937
} 
#line 939
static inline unsigned long min(const unsigned long a, const long b) 
#line 940
{ 
#line 941
unsigned long retval; 
#line 943
#pragma warning (push)
#pragma warning (disable: 4127)
#line 946 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 948
#pragma warning (pop)
#line 950 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 951
} else { 
#line 952
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 953
}  
#line 954
return retval; 
#line 955
} 
#line 957
static inline __int64 min(const __int64 a, const __int64 b) 
#line 958
{ 
#line 959
return llmin(a, b); 
#line 960
} 
#line 962
static inline unsigned __int64 min(const unsigned __int64 a, const unsigned __int64 b) 
#line 963
{ 
#line 964
return ullmin(a, b); 
#line 965
} 
#line 967
static inline unsigned __int64 min(const __int64 a, const unsigned __int64 b) 
#line 968
{ 
#line 969
return ullmin(static_cast< unsigned __int64>(a), b); 
#line 970
} 
#line 972
static inline unsigned __int64 min(const unsigned __int64 a, const __int64 b) 
#line 973
{ 
#line 974
return ullmin(a, static_cast< unsigned __int64>(b)); 
#line 975
} 
#line 977
static inline float min(const float a, const float b) 
#line 978
{ 
#line 979
return fminf(a, b); 
#line 980
} 
#line 982
static inline double min(const double a, const double b) 
#line 983
{ 
#line 984
return fmin(a, b); 
#line 985
} 
#line 987
static inline double min(const float a, const double b) 
#line 988
{ 
#line 989
return fmin(static_cast< double>(a), b); 
#line 990
} 
#line 992
static inline double min(const double a, const float b) 
#line 993
{ 
#line 994
return fmin(a, static_cast< double>(b)); 
#line 995
} 
#line 997
static inline unsigned max(const unsigned a, const unsigned b) 
#line 998
{ 
#line 999
return umax(a, b); 
#line 1000
} 
#line 1002
static inline unsigned max(const int a, const unsigned b) 
#line 1003
{ 
#line 1004
return umax(static_cast< unsigned>(a), b); 
#line 1005
} 
#line 1007
static inline unsigned max(const unsigned a, const int b) 
#line 1008
{ 
#line 1009
return umax(a, static_cast< unsigned>(b)); 
#line 1010
} 
#line 1012
static inline long max(const long a, const long b) 
#line 1013
{ 
#line 1014
long retval; 
#line 1017
#pragma warning (push)
#pragma warning (disable: 4127)
#line 1020 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 1022
#pragma warning (pop)
#line 1024 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< long>(max(static_cast< int>(a), static_cast< int>(b)))); 
#line 1025
} else { 
#line 1026
retval = (static_cast< long>(llmax(static_cast< __int64>(a), static_cast< __int64>(b)))); 
#line 1027
}  
#line 1028
return retval; 
#line 1029
} 
#line 1031
static inline unsigned long max(const unsigned long a, const unsigned long b) 
#line 1032
{ 
#line 1033
unsigned long retval; 
#line 1035
#pragma warning (push)
#pragma warning (disable: 4127)
#line 1038 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1040
#pragma warning (pop)
#line 1042 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1043
} else { 
#line 1044
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1045
}  
#line 1046
return retval; 
#line 1047
} 
#line 1049
static inline unsigned long max(const long a, const unsigned long b) 
#line 1050
{ 
#line 1051
unsigned long retval; 
#line 1053
#pragma warning (push)
#pragma warning (disable: 4127)
#line 1056 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1058
#pragma warning (pop)
#line 1060 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1061
} else { 
#line 1062
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1063
}  
#line 1064
return retval; 
#line 1065
} 
#line 1067
static inline unsigned long max(const unsigned long a, const long b) 
#line 1068
{ 
#line 1069
unsigned long retval; 
#line 1071
#pragma warning (push)
#pragma warning (disable: 4127)
#line 1074 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1076
#pragma warning (pop)
#line 1078 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1079
} else { 
#line 1080
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1081
}  
#line 1082
return retval; 
#line 1083
} 
#line 1085
static inline __int64 max(const __int64 a, const __int64 b) 
#line 1086
{ 
#line 1087
return llmax(a, b); 
#line 1088
} 
#line 1090
static inline unsigned __int64 max(const unsigned __int64 a, const unsigned __int64 b) 
#line 1091
{ 
#line 1092
return ullmax(a, b); 
#line 1093
} 
#line 1095
static inline unsigned __int64 max(const __int64 a, const unsigned __int64 b) 
#line 1096
{ 
#line 1097
return ullmax(static_cast< unsigned __int64>(a), b); 
#line 1098
} 
#line 1100
static inline unsigned __int64 max(const unsigned __int64 a, const __int64 b) 
#line 1101
{ 
#line 1102
return ullmax(a, static_cast< unsigned __int64>(b)); 
#line 1103
} 
#line 1105
static inline float max(const float a, const float b) 
#line 1106
{ 
#line 1107
return fmaxf(a, b); 
#line 1108
} 
#line 1110
static inline double max(const double a, const double b) 
#line 1111
{ 
#line 1112
return fmax(a, b); 
#line 1113
} 
#line 1115
static inline double max(const float a, const double b) 
#line 1116
{ 
#line 1117
return fmax(static_cast< double>(a), b); 
#line 1118
} 
#line 1120
static inline double max(const double a, const float b) 
#line 1121
{ 
#line 1122
return fmax(a, static_cast< double>(b)); 
#line 1123
} 
#line 1129
#pragma warning (push)
#pragma warning (disable : 4211)
#line 1135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\math_functions.hpp"
static inline int min(const int a, const int b) 
#line 1136
{ 
#line 1137
return (a < b) ? a : b; 
#line 1138
} 
#line 1140
static inline unsigned umin(const unsigned a, const unsigned b) 
#line 1141
{ 
#line 1142
return (a < b) ? a : b; 
#line 1143
} 
#line 1145
static inline __int64 llmin(const __int64 a, const __int64 b) 
#line 1146
{ 
#line 1147
return (a < b) ? a : b; 
#line 1148
} 
#line 1150
static inline unsigned __int64 ullmin(const unsigned __int64 a, const unsigned __int64 
#line 1151
b) 
#line 1152
{ 
#line 1153
return (a < b) ? a : b; 
#line 1154
} 
#line 1156
static inline int max(const int a, const int b) 
#line 1157
{ 
#line 1158
return (a > b) ? a : b; 
#line 1159
} 
#line 1161
static inline unsigned umax(const unsigned a, const unsigned b) 
#line 1162
{ 
#line 1163
return (a > b) ? a : b; 
#line 1164
} 
#line 1166
static inline __int64 llmax(const __int64 a, const __int64 b) 
#line 1167
{ 
#line 1168
return (a > b) ? a : b; 
#line 1169
} 
#line 1171
static inline unsigned __int64 ullmax(const unsigned __int64 a, const unsigned __int64 
#line 1172
b) 
#line 1173
{ 
#line 1174
return (a > b) ? a : b; 
#line 1175
} 
#line 1178
#pragma warning (pop)
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt/device_functions.h"
extern "C" {
#line 2497 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt/device_functions.h"
static __inline int __vimax_s32_relu(const int a, const int b); 
#line 2509
static __inline unsigned __vimax_s16x2_relu(const unsigned a, const unsigned b); 
#line 2518
static __inline int __vimin_s32_relu(const int a, const int b); 
#line 2530
static __inline unsigned __vimin_s16x2_relu(const unsigned a, const unsigned b); 
#line 2539
static __inline int __vimax3_s32(const int a, const int b, const int c); 
#line 2551
static __inline unsigned __vimax3_s16x2(const unsigned a, const unsigned b, const unsigned c); 
#line 2560
static __inline unsigned __vimax3_u32(const unsigned a, const unsigned b, const unsigned c); 
#line 2572
static __inline unsigned __vimax3_u16x2(const unsigned a, const unsigned b, const unsigned c); 
#line 2581
static __inline int __vimin3_s32(const int a, const int b, const int c); 
#line 2593
static __inline unsigned __vimin3_s16x2(const unsigned a, const unsigned b, const unsigned c); 
#line 2602
static __inline unsigned __vimin3_u32(const unsigned a, const unsigned b, const unsigned c); 
#line 2614
static __inline unsigned __vimin3_u16x2(const unsigned a, const unsigned b, const unsigned c); 
#line 2623
static __inline int __vimax3_s32_relu(const int a, const int b, const int c); 
#line 2635
static __inline unsigned __vimax3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
#line 2644
static __inline int __vimin3_s32_relu(const int a, const int b, const int c); 
#line 2656
static __inline unsigned __vimin3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
#line 2665
static __inline int __viaddmax_s32(const int a, const int b, const int c); 
#line 2677
static __inline unsigned __viaddmax_s16x2(const unsigned a, const unsigned b, const unsigned c); 
#line 2686
static __inline unsigned __viaddmax_u32(const unsigned a, const unsigned b, const unsigned c); 
#line 2698
static __inline unsigned __viaddmax_u16x2(const unsigned a, const unsigned b, const unsigned c); 
#line 2707
static __inline int __viaddmin_s32(const int a, const int b, const int c); 
#line 2719
static __inline unsigned __viaddmin_s16x2(const unsigned a, const unsigned b, const unsigned c); 
#line 2728
static __inline unsigned __viaddmin_u32(const unsigned a, const unsigned b, const unsigned c); 
#line 2740
static __inline unsigned __viaddmin_u16x2(const unsigned a, const unsigned b, const unsigned c); 
#line 2750
static __inline int __viaddmax_s32_relu(const int a, const int b, const int c); 
#line 2762
static __inline unsigned __viaddmax_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
#line 2772
static __inline int __viaddmin_s32_relu(const int a, const int b, const int c); 
#line 2784
static __inline unsigned __viaddmin_s16x2_relu(const unsigned a, const unsigned b, const unsigned c); 
#line 2793
static __inline int __vibmax_s32(const int a, const int b, bool *const pred); 
#line 2802
static __inline unsigned __vibmax_u32(const unsigned a, const unsigned b, bool *const pred); 
#line 2811
static __inline int __vibmin_s32(const int a, const int b, bool *const pred); 
#line 2820
static __inline unsigned __vibmin_u32(const unsigned a, const unsigned b, bool *const pred); 
#line 2834
static __inline unsigned __vibmax_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
#line 2848
static __inline unsigned __vibmax_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
#line 2862
static __inline unsigned __vibmin_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
#line 2876
static __inline unsigned __vibmin_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo); 
#line 2883
}
#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
static short __internal_cast_u2s(unsigned short x) 
#line 117
{ 
#line 118
short res; 
#line 120
(void)memcpy(&res, &x, sizeof x); 
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 125
} 
#line 127
static __inline int __vimax_s32_relu(const int a, const int b) { 
#line 134
int ans = max(a, b); 
#line 136
return (ans > 0) ? ans : 0; 
#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 140
static __inline unsigned __vimax_s16x2_relu(const unsigned a, const unsigned b) { 
#line 141
unsigned res; 
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 150
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 152
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 153
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 156
short aS_lo = __internal_cast_u2s(aU_lo); 
#line 157
short aS_hi = __internal_cast_u2s(aU_hi); 
#line 159
short bS_lo = __internal_cast_u2s(bU_lo); 
#line 160
short bS_hi = __internal_cast_u2s(bU_hi); 
#line 163
int ansI_lo = max(aS_lo, bS_lo); 
#line 164
int ansI_hi = max(aS_hi, bS_hi); 
#line 167
if (ansI_lo < 0) { ansI_lo = 0; }  
#line 168
if (ansI_hi < 0) { ansI_hi = 0; }  
#line 171
unsigned ansU_lo = (unsigned)ansI_lo; 
#line 172
unsigned ansU_hi = (unsigned)ansI_hi; 
#line 175
res = (ansU_lo | (ansU_hi << 16)); 
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 179
} 
#line 181
static __inline int __vimin_s32_relu(const int a, const int b) { 
#line 188
int ans = min(a, b); 
#line 190
return (ans > 0) ? ans : 0; 
#line 192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 194
static __inline unsigned __vimin_s16x2_relu(const unsigned a, const unsigned b) { 
#line 195
unsigned res; 
#line 203 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 204
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 206
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 207
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 210
short aS_lo = __internal_cast_u2s(aU_lo); 
#line 211
short aS_hi = __internal_cast_u2s(aU_hi); 
#line 213
short bS_lo = __internal_cast_u2s(bU_lo); 
#line 214
short bS_hi = __internal_cast_u2s(bU_hi); 
#line 217
int ansI_lo = min(aS_lo, bS_lo); 
#line 218
int ansI_hi = min(aS_hi, bS_hi); 
#line 221
if (ansI_lo < 0) { ansI_lo = 0; }  
#line 222
if (ansI_hi < 0) { ansI_hi = 0; }  
#line 225
unsigned ansU_lo = (unsigned)ansI_lo; 
#line 226
unsigned ansU_hi = (unsigned)ansI_hi; 
#line 229
res = (ansU_lo | (ansU_hi << 16)); 
#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 233
} 
#line 235
static __inline int __vimax3_s32(const int a, const int b, const int c) { 
#line 245
return max(max(a, b), c); 
#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 249
static __inline unsigned __vimax3_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
#line 250
unsigned res; 
#line 262 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 263
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 265
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 266
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 268
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 269
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 272
short aS_lo = __internal_cast_u2s(aU_lo); 
#line 273
short aS_hi = __internal_cast_u2s(aU_hi); 
#line 275
short bS_lo = __internal_cast_u2s(bU_lo); 
#line 276
short bS_hi = __internal_cast_u2s(bU_hi); 
#line 278
short cS_lo = __internal_cast_u2s(cU_lo); 
#line 279
short cS_hi = __internal_cast_u2s(cU_hi); 
#line 282
unsigned ansU_lo = (unsigned)max(max(aS_lo, bS_lo), cS_lo); 
#line 283
unsigned ansU_hi = (unsigned)max(max(aS_hi, bS_hi), cS_hi); 
#line 286
res = ((ansU_lo & 65535U) | (ansU_hi << 16)); 
#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 289
} 
#line 291
static __inline unsigned __vimax3_u32(const unsigned a, const unsigned b, const unsigned c) { 
#line 301
return max(max(a, b), c); 
#line 303 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 305
static __inline unsigned __vimax3_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
#line 306
unsigned res; 
#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 318
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 320
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 321
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 323
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 324
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 327
unsigned short ansU_lo = (unsigned short)max(max(aU_lo, bU_lo), cU_lo); 
#line 328
unsigned short ansU_hi = (unsigned short)max(max(aU_hi, bU_hi), cU_hi); 
#line 331
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
#line 334 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 335
} 
#line 337
static __inline int __vimin3_s32(const int a, const int b, const int c) { 
#line 347
return min(min(a, b), c); 
#line 349 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 351
static __inline unsigned __vimin3_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
#line 352
unsigned res; 
#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 364
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 366
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 367
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 369
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 370
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 373
short aS_lo = __internal_cast_u2s(aU_lo); 
#line 374
short aS_hi = __internal_cast_u2s(aU_hi); 
#line 376
short bS_lo = __internal_cast_u2s(bU_lo); 
#line 377
short bS_hi = __internal_cast_u2s(bU_hi); 
#line 379
short cS_lo = __internal_cast_u2s(cU_lo); 
#line 380
short cS_hi = __internal_cast_u2s(cU_hi); 
#line 383
unsigned ansU_lo = (unsigned)min(min(aS_lo, bS_lo), cS_lo); 
#line 384
unsigned ansU_hi = (unsigned)min(min(aS_hi, bS_hi), cS_hi); 
#line 387
res = ((ansU_lo & 65535U) | (ansU_hi << 16)); 
#line 390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 391
} 
#line 393
static __inline unsigned __vimin3_u32(const unsigned a, const unsigned b, const unsigned c) { 
#line 403
return min(min(a, b), c); 
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 407
static __inline unsigned __vimin3_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
#line 408
unsigned res; 
#line 419 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 420
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 422
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 423
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 425
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 426
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 429
unsigned short ansU_lo = (unsigned short)min(min(aU_lo, bU_lo), cU_lo); 
#line 430
unsigned short ansU_hi = (unsigned short)min(min(aU_hi, bU_hi), cU_hi); 
#line 433
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
#line 436 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 437
} 
#line 439
static __inline int __vimax3_s32_relu(const int a, const int b, const int c) { 
#line 449
int ans = max(max(a, b), c); 
#line 451
return (ans > 0) ? ans : 0; 
#line 453 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 455
static __inline unsigned __vimax3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
#line 456
unsigned res; 
#line 467 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 468
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 470
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 471
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 473
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 474
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 477
short aS_lo = __internal_cast_u2s(aU_lo); 
#line 478
short aS_hi = __internal_cast_u2s(aU_hi); 
#line 480
short bS_lo = __internal_cast_u2s(bU_lo); 
#line 481
short bS_hi = __internal_cast_u2s(bU_hi); 
#line 483
short cS_lo = __internal_cast_u2s(cU_lo); 
#line 484
short cS_hi = __internal_cast_u2s(cU_hi); 
#line 487
unsigned ansU_lo = (unsigned)max(0, max(max(aS_lo, bS_lo), cS_lo)); 
#line 488
unsigned ansU_hi = (unsigned)max(0, max(max(aS_hi, bS_hi), cS_hi)); 
#line 491
res = (ansU_lo | (ansU_hi << 16)); 
#line 494 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 495
} 
#line 497
static __inline int __vimin3_s32_relu(const int a, const int b, const int c) { 
#line 507
int ans = min(min(a, b), c); 
#line 509
return (ans > 0) ? ans : 0; 
#line 511 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 513
static __inline unsigned __vimin3_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
#line 514
unsigned res; 
#line 525 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 526
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 528
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 529
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 531
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 532
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 535
short aS_lo = __internal_cast_u2s(aU_lo); 
#line 536
short aS_hi = __internal_cast_u2s(aU_hi); 
#line 538
short bS_lo = __internal_cast_u2s(bU_lo); 
#line 539
short bS_hi = __internal_cast_u2s(bU_hi); 
#line 541
short cS_lo = __internal_cast_u2s(cU_lo); 
#line 542
short cS_hi = __internal_cast_u2s(cU_hi); 
#line 545
unsigned ansU_lo = (unsigned)max(0, min(min(aS_lo, bS_lo), cS_lo)); 
#line 546
unsigned ansU_hi = (unsigned)max(0, min(min(aS_hi, bS_hi), cS_hi)); 
#line 549
res = (ansU_lo | (ansU_hi << 16)); 
#line 553 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 554
} 
#line 556
static __inline int __viaddmax_s32(const int a, const int b, const int c) { 
#line 566
return max(a + b, c); 
#line 568 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 570
static __inline unsigned __viaddmax_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
#line 571
unsigned res; 
#line 582 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 583
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 585
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 586
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 588
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 589
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 591
aU_lo += bU_lo; 
#line 592
aU_hi += bU_hi; 
#line 595
short sS_lo = __internal_cast_u2s(aU_lo); 
#line 596
short sS_hi = __internal_cast_u2s(aU_hi); 
#line 598
short cS_lo = __internal_cast_u2s(cU_lo); 
#line 599
short cS_hi = __internal_cast_u2s(cU_hi); 
#line 602
unsigned ansU_lo = (unsigned)max(sS_lo, cS_lo); 
#line 603
unsigned ansU_hi = (unsigned)max(sS_hi, cS_hi); 
#line 606
res = ((ansU_lo & 65535U) | (ansU_hi << 16)); 
#line 609 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 610
} 
#line 612
static __inline unsigned __viaddmax_u32(const unsigned a, const unsigned b, const unsigned c) { 
#line 622
return max(a + b, c); 
#line 624 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 626
static __inline unsigned __viaddmax_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
#line 627
unsigned res; 
#line 638 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 639
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 641
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 642
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 644
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 645
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 648
unsigned short ansU_lo = (unsigned short)max((unsigned short)(aU_lo + bU_lo), cU_lo); 
#line 649
unsigned short ansU_hi = (unsigned short)max((unsigned short)(aU_hi + bU_hi), cU_hi); 
#line 652
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
#line 655 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 656
} 
#line 658
static __inline int __viaddmin_s32(const int a, const int b, const int c) { 
#line 668
return min(a + b, c); 
#line 670 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 672
static __inline unsigned __viaddmin_s16x2(const unsigned a, const unsigned b, const unsigned c) { 
#line 673
unsigned res; 
#line 684 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 685
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 687
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 688
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 690
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 691
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 693
aU_lo += bU_lo; 
#line 694
aU_hi += bU_hi; 
#line 697
short sS_lo = __internal_cast_u2s(aU_lo); 
#line 698
short sS_hi = __internal_cast_u2s(aU_hi); 
#line 700
short cS_lo = __internal_cast_u2s(cU_lo); 
#line 701
short cS_hi = __internal_cast_u2s(cU_hi); 
#line 704
unsigned ansU_lo = (unsigned)min(sS_lo, cS_lo); 
#line 705
unsigned ansU_hi = (unsigned)min(sS_hi, cS_hi); 
#line 708
res = ((ansU_lo & 65535U) | (ansU_hi << 16)); 
#line 711 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 712
} 
#line 714
static __inline unsigned __viaddmin_u32(const unsigned a, const unsigned b, const unsigned c) { 
#line 724
return min(a + b, c); 
#line 726 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 728
static __inline unsigned __viaddmin_u16x2(const unsigned a, const unsigned b, const unsigned c) { 
#line 729
unsigned res; 
#line 740 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 741
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 743
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 744
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 746
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 747
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 750
unsigned short ansU_lo = (unsigned short)min((unsigned short)(aU_lo + bU_lo), cU_lo); 
#line 751
unsigned short ansU_hi = (unsigned short)min((unsigned short)(aU_hi + bU_hi), cU_hi); 
#line 754
res = (((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16)); 
#line 757 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 758
} 
#line 760
static __inline int __viaddmax_s32_relu(const int a, const int b, const int c) { 
#line 770
int ans = max(a + b, c); 
#line 772
return (ans > 0) ? ans : 0; 
#line 774 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 776
static __inline unsigned __viaddmax_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
#line 777
unsigned res; 
#line 788 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 789
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 791
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 792
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 794
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 795
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 797
aU_lo += bU_lo; 
#line 798
aU_hi += bU_hi; 
#line 801
short sS_lo = __internal_cast_u2s(aU_lo); 
#line 802
short sS_hi = __internal_cast_u2s(aU_hi); 
#line 804
short cS_lo = __internal_cast_u2s(cU_lo); 
#line 805
short cS_hi = __internal_cast_u2s(cU_hi); 
#line 808
unsigned ansU_lo = (unsigned)max(0, max(sS_lo, cS_lo)); 
#line 809
unsigned ansU_hi = (unsigned)max(0, max(sS_hi, cS_hi)); 
#line 812
res = (ansU_lo | (ansU_hi << 16)); 
#line 815 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 816
} 
#line 818
static __inline int __viaddmin_s32_relu(const int a, const int b, const int c) { 
#line 828
int ans = min(a + b, c); 
#line 830
return (ans > 0) ? ans : 0; 
#line 832 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 834
static __inline unsigned __viaddmin_s16x2_relu(const unsigned a, const unsigned b, const unsigned c) { 
#line 835
unsigned res; 
#line 846 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 847
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 849
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 850
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 852
unsigned short cU_lo = (unsigned short)(c & 65535U); 
#line 853
unsigned short cU_hi = (unsigned short)(c >> 16); 
#line 855
aU_lo += bU_lo; 
#line 856
aU_hi += bU_hi; 
#line 859
short sS_lo = __internal_cast_u2s(aU_lo); 
#line 860
short sS_hi = __internal_cast_u2s(aU_hi); 
#line 862
short cS_lo = __internal_cast_u2s(cU_lo); 
#line 863
short cS_hi = __internal_cast_u2s(cU_hi); 
#line 866
unsigned ansU_lo = (unsigned)max(0, min(sS_lo, cS_lo)); 
#line 867
unsigned ansU_hi = (unsigned)max(0, min(sS_hi, cS_hi)); 
#line 870
res = (ansU_lo | (ansU_hi << 16)); 
#line 873 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
return res; 
#line 874
} 
#line 878
static __inline int __vibmax_s32(const int a, const int b, bool *const pred) { 
#line 892
int ans = max(a, b); 
#line 894
(*pred) = (a >= b); 
#line 895
return ans; 
#line 897 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 899
static __inline unsigned __vibmax_u32(const unsigned a, const unsigned b, bool *const pred) { 
#line 913
unsigned ans = max(a, b); 
#line 915
(*pred) = (a >= b); 
#line 916
return ans; 
#line 918 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 921
static __inline int __vibmin_s32(const int a, const int b, bool *const pred) { 
#line 935
int ans = min(a, b); 
#line 937
(*pred) = (a <= b); 
#line 938
return ans; 
#line 940 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 943
static __inline unsigned __vibmin_u32(const unsigned a, const unsigned b, bool *const pred) { 
#line 957
unsigned ans = min(a, b); 
#line 959
(*pred) = (a <= b); 
#line 960
return ans; 
#line 962 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 964
static __inline unsigned __vibmax_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
#line 986
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 987
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 989
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 990
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 993
short aS_lo = __internal_cast_u2s(aU_lo); 
#line 994
short aS_hi = __internal_cast_u2s(aU_hi); 
#line 996
short bS_lo = __internal_cast_u2s(bU_lo); 
#line 997
short bS_hi = __internal_cast_u2s(bU_hi); 
#line 1000
unsigned ansU_lo = (unsigned)max(aS_lo, bS_lo); 
#line 1001
unsigned ansU_hi = (unsigned)max(aS_hi, bS_hi); 
#line 1003
(*pred_hi) = (aS_hi >= bS_hi); 
#line 1004
(*pred_lo) = (aS_lo >= bS_lo); 
#line 1007
unsigned ans = (ansU_lo & 65535U) | (ansU_hi << 16); 
#line 1009
return ans; 
#line 1011 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 1013
static __inline unsigned __vibmax_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
#line 1035
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 1036
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 1038
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 1039
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 1042
unsigned short ansU_lo = (unsigned short)max(aU_lo, bU_lo); 
#line 1043
unsigned short ansU_hi = (unsigned short)max(aU_hi, bU_hi); 
#line 1045
(*pred_hi) = (aU_hi >= bU_hi); 
#line 1046
(*pred_lo) = (aU_lo >= bU_lo); 
#line 1049
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
#line 1051
return ans; 
#line 1053 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 1055
static __inline unsigned __vibmin_s16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
#line 1077
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 1078
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 1080
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 1081
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 1084
short aS_lo = __internal_cast_u2s(aU_lo); 
#line 1085
short aS_hi = __internal_cast_u2s(aU_hi); 
#line 1087
short bS_lo = __internal_cast_u2s(bU_lo); 
#line 1088
short bS_hi = __internal_cast_u2s(bU_hi); 
#line 1091
unsigned ansU_lo = (unsigned)min(aS_lo, bS_lo); 
#line 1092
unsigned ansU_hi = (unsigned)min(aS_hi, bS_hi); 
#line 1094
(*pred_hi) = (aS_hi <= bS_hi); 
#line 1095
(*pred_lo) = (aS_lo <= bS_lo); 
#line 1098
unsigned ans = (ansU_lo & 65535U) | (ansU_hi << 16); 
#line 1100
return ans; 
#line 1102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
#line 1104
static __inline unsigned __vibmin_u16x2(const unsigned a, const unsigned b, bool *const pred_hi, bool *const pred_lo) { 
#line 1126
unsigned short aU_lo = (unsigned short)(a & 65535U); 
#line 1127
unsigned short aU_hi = (unsigned short)(a >> 16); 
#line 1129
unsigned short bU_lo = (unsigned short)(b & 65535U); 
#line 1130
unsigned short bU_hi = (unsigned short)(b >> 16); 
#line 1133
unsigned short ansU_lo = (unsigned short)min(aU_lo, bU_lo); 
#line 1134
unsigned short ansU_hi = (unsigned short)min(aU_hi, bU_hi); 
#line 1136
(*pred_hi) = (aU_hi <= bU_hi); 
#line 1137
(*pred_lo) = (aU_lo <= bU_lo); 
#line 1140
unsigned ans = ((unsigned)ansU_lo) | (((unsigned)ansU_hi) << 16); 
#line 1142
return ans; 
#line 1144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_functions.hpp"
} 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicAdd(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 89
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicAdd(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 91
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicSub(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 93
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicSub(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 95
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicExch(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 97
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicExch(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 99
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline float atomicExch(float *address, float val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 101
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicMin(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 103
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicMin(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 105
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicMax(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 107
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicMax(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 109
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicInc(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 111
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicDec(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 113
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicAnd(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 115
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicAnd(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 117
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicOr(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 119
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 121 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicOr(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 121
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicXor(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 123
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicXor(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 125
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline int atomicCAS(int *address, int compare, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 127
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 156 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
extern "C" {
#line 160
}
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 169
static __inline unsigned __int64 atomicAdd(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 169
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicExch(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 171
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicCAS(unsigned __int64 *address, unsigned __int64 compare, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 173
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
__declspec(deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning).")) static __inline bool any(bool cond) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 175
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_atomic_functions.h"
__declspec(deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning).")) static __inline bool all(bool cond) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 177
{ } 
#endif
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.h"
extern "C" {
#line 1142
}
#line 1150
static __inline double fma(double a, double b, double c, cudaRoundMode mode); 
#line 1154
static __inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1156
static __inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1158
static __inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1160
static __inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1162
static __inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1164
static __inline __int64 double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1166
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1168
static __inline double ll2double(__int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1170
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1172
static __inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
#line 1174
static __inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
#line 1176
static __inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double fma(double a, double b, double c, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;::exit(___);}
#if 0
#line 94
{ 
#line 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
#line 99
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double dmul(double a, double b, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 102
{ 
#line 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
#line 107
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double dadd(double a, double b, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 110
{ 
#line 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
#line 115
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double dsub(double a, double b, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 118
{ 
#line 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
#line 123
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline int double2int(double a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 126
{ 
#line 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
#line 131
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline unsigned double2uint(double a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 134
{ 
#line 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
#line 139
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline __int64 double2ll(double a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 142
{ 
#line 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
#line 147
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 150
{ 
#line 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
#line 155
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double ll2double(__int64 a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 158
{ 
#line 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
#line 163
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 166
{ 
#line 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
#line 171
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double int2double(int a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 174
{ 
#line 175
return (double)a; 
#line 176
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double uint2double(unsigned a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 179
{ 
#line 180
return (double)a; 
#line 181
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\device_double_functions.hpp"
static __inline double float2double(float a, cudaRoundMode mode) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 184
{ 
#line 185
return (double)a; 
#line 186
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_atomic_functions.h"
static __inline float atomicAdd(float *address, float val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 88
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMin(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 89
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMax(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 91
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicAnd(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 93
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicOr(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 95
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicXor(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 97
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMin(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 99
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMax(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 101
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicAnd(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 103
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicOr(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 105
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicXor(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 107
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd(double *address, double val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 93
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_block(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 96
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_system(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 99
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 102
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 105
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_block(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 108
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_system(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 111
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_block(float *address, float val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 114
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_system(float *address, float val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 117
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_block(double *address, double val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 120
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_system(double *address, double val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 123
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicSub_block(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 126
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicSub_system(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 129
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicSub_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 132
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicSub_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 135
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_block(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 138
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_system(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 141
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 144
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 147 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 147
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 150 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_block(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 150
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_system(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 153
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 156 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_block(float *address, float val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 156
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_system(float *address, float val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 159
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_block(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 162
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_system(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 165
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_block(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 168
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_system(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 171
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 174
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 177
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_block(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 180
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_system(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 183
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 186 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_block(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 186
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_system(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 189
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_block(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 192
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 195 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_system(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 195
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 198 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 198
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 201 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 201
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_block(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 204
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_system(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 207
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 210 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 210
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 213
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 216
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 219
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 222 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_block(int *address, int compare, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 222
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 225 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_system(int *address, int compare, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 225
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 228 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
#line 229
val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 229
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
#line 233
val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 233
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 236 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_block(unsigned __int64 *address, unsigned __int64 
#line 237
compare, unsigned __int64 
#line 238
val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 238
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 241 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_system(unsigned __int64 *address, unsigned __int64 
#line 242
compare, unsigned __int64 
#line 243
val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 243
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 246 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_block(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 246
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 249 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_system(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 249
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_block(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 252
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 255 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_system(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 255
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 258 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 258
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 261 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 261
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 264 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_block(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 264
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_system(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 267
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 270 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_block(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 270
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 273 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_system(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 273
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_block(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 276
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_system(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 279
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 282 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 282
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 285
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_block(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 288
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 291 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_system(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 291
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 294 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_block(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 294
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 297 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_system(int *address, int val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 297
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 300 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_block(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 300
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 303 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_system(__int64 *address, __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 303
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_block(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 306
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_system(unsigned *address, unsigned val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 309
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_block(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 312
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_system(unsigned __int64 *address, unsigned __int64 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 315
{ } 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
extern "C" {
#line 1038
}
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1045
__declspec(deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning).")) static __inline unsigned ballot(bool pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1045
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1047 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline int syncthreads_count(bool pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1047
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1049 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_and(bool pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1049
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1051 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_or(bool pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1051
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1056 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isGlobal(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1056
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1057 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isShared(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1057
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1058 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isConstant(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1058
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1059 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isLocal(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1059
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1061 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isGridConstant(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1061
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1063 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_global(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1063
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1064 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_shared(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1064
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1065 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_constant(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1065
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1066 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_local(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1066
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1068 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_grid_constant(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1068
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1071 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_global_to_generic(size_t rawbits) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1071
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1072 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_shared_to_generic(size_t rawbits) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1072
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1073 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_constant_to_generic(size_t rawbits) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1073
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1074 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_local_to_generic(size_t rawbits) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1074
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 1076 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_grid_constant_to_generic(size_t rawbits) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1076
{ } 
#endif
#line 1090 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
static __inline unsigned short __nv_bswap16(unsigned short x) { 
#line 1094 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
return _byteswap_ushort(x); 
#line 1098 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
} 
#line 1109
static __inline unsigned __nv_bswap32(unsigned x) { 
#line 1113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
unsigned long ret = _byteswap_ulong(static_cast< unsigned long>(x)); 
#line 1114
return static_cast< unsigned>(ret); 
#line 1118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
} 
#line 1129
static __inline unsigned __int64 __nv_bswap64(unsigned __int64 x) { 
#line 1133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
unsigned __int64 ret = _byteswap_uint64(x); 
#line 1134
return ret; 
#line 1138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_20_intrinsics.h"
} 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __fns(unsigned mask, unsigned base, int offset) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
#line 123
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline void __barrier_sync(unsigned id) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)id;::exit(___);}
#if 0
#line 124
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline void __barrier_sync_count(unsigned id, unsigned cnt) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
#line 125
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline void __syncwarp(unsigned mask = 4294967295U) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
#line 126
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline int __all_sync(unsigned mask, int pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 127
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline int __any_sync(unsigned mask, int pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 128
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline int __uni_sync(unsigned mask, int pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 129
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __ballot_sync(unsigned mask, int pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 130
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 131 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __activemask() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 131
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline int __shfl(int var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 140
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned __shfl(unsigned var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 141
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline int __shfl_up(int var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 142
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 143 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 143
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline int __shfl_down(int var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 144
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 145
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline int __shfl_xor(int var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 146
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 147 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 147
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 148 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline float __shfl(float var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 148
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline float __shfl_up(float var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 149
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 150 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline float __shfl_down(float var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 150
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline float __shfl_xor(float var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 151
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 154
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 155 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 155
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 156 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 156
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 157
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 158 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 158
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 159
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 160 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 160
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 161 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 161
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 162
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 163 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 163
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 164
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 165
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned __int64 __shfl(unsigned __int64 var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 169
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline __int64 __shfl(__int64 var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 170
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline __int64 __shfl_up(__int64 var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 171
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned __int64 __shfl_up(unsigned __int64 var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 172
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline __int64 __shfl_down(__int64 var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 173
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned __int64 __shfl_down(unsigned __int64 var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 174
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline __int64 __shfl_xor(__int64 var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 175
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned __int64 __shfl_xor(unsigned __int64 var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 176
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline double __shfl(double var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 177
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline double __shfl_up(double var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 178
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline double __shfl_down(double var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 179
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline double __shfl_xor(double var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 180
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_sync(unsigned mask, __int64 var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 183
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_sync(unsigned mask, unsigned __int64 var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 184
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_up_sync(unsigned mask, __int64 var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 185
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 186 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_up_sync(unsigned mask, unsigned __int64 var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 186
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 187 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_down_sync(unsigned mask, __int64 var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 187
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_down_sync(unsigned mask, unsigned __int64 var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 188
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_xor_sync(unsigned mask, __int64 var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 189
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_xor_sync(unsigned mask, unsigned __int64 var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 190
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 191
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 192
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 193 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 193
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 194 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 194
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 198 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline long __shfl(long var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 198
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 199 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 199
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 200 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline long __shfl_up(long var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 200
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 201 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 201
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 202 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline long __shfl_down(long var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 202
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 203 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 203
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline long __shfl_xor(long var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 204
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 205 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 205
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 208 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 208
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 209 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 209
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 210 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 210
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 211 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 211
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 212
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 213
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 214 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 214
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 215 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 215
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline long __ldg(const long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 91
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 92 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldg(const unsigned long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 92
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char __ldg(const char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 94
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldg(const signed char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 95
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short __ldg(const short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 96
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int __ldg(const int *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 97
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldg(const __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 98
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldg(const char2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 99
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldg(const char4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 100
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldg(const short2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 101
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldg(const short4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 102
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldg(const int2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 103
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldg(const int4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 104
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldg(const longlong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 105
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldg(const unsigned char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 107
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldg(const unsigned short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 108
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldg(const unsigned *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 109
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldg(const unsigned __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 110
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldg(const uchar2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 111
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldg(const uchar4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 112
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldg(const ushort2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 113
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldg(const ushort4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 114
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldg(const uint2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 115
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldg(const uint4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 116
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldg(const ulonglong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 117
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float __ldg(const float *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 119
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double __ldg(const double *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 120
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 121 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldg(const float2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 121
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldg(const float4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 122
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldg(const double2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 123
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline long __ldcg(const long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 128
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcg(const unsigned long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 129
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 131 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char __ldcg(const char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 131
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcg(const signed char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 132
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short __ldcg(const short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 133
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int __ldcg(const int *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 134
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcg(const __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 135
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcg(const char2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 136
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 137 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcg(const char4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 137
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcg(const short2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 138
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 139 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcg(const short4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 139
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcg(const int2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 140
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcg(const int4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 141
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcg(const longlong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 142
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcg(const unsigned char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 144
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcg(const unsigned short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 145
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcg(const unsigned *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 146
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 147 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcg(const unsigned __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 147
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 148 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcg(const uchar2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 148
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcg(const uchar4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 149
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 150 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcg(const ushort2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 150
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcg(const ushort4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 151
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 152 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcg(const uint2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 152
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcg(const uint4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 153
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcg(const ulonglong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 154
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 156 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float __ldcg(const float *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 156
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double __ldcg(const double *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 157
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 158 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcg(const float2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 158
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcg(const float4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 159
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 160 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcg(const double2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 160
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline long __ldca(const long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 164
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldca(const unsigned long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 165
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 167 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char __ldca(const char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 167
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldca(const signed char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 168
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short __ldca(const short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 169
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int __ldca(const int *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 170
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldca(const __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 171
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldca(const char2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 172
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldca(const char4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 173
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldca(const short2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 174
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldca(const short4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 175
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldca(const int2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 176
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldca(const int4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 177
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldca(const longlong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 178
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldca(const unsigned char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 180
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 181 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldca(const unsigned short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 181
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 182 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldca(const unsigned *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 182
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldca(const unsigned __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 183
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldca(const uchar2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 184
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldca(const uchar4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 185
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 186 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldca(const ushort2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 186
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 187 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldca(const ushort4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 187
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldca(const uint2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 188
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldca(const uint4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 189
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldca(const ulonglong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 190
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float __ldca(const float *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 192
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 193 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double __ldca(const double *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 193
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 194 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldca(const float2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 194
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 195 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldca(const float4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 195
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldca(const double2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 196
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 200 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline long __ldcs(const long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 200
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 201 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcs(const unsigned long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 201
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 203 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char __ldcs(const char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 203
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcs(const signed char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 204
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 205 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short __ldcs(const short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 205
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 206 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int __ldcs(const int *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 206
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcs(const __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 207
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 208 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcs(const char2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 208
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 209 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcs(const char4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 209
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 210 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcs(const short2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 210
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 211 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcs(const short4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 211
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcs(const int2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 212
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcs(const int4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 213
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 214 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcs(const longlong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 214
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcs(const unsigned char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 216
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 217 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcs(const unsigned short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 217
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 218 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcs(const unsigned *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 218
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcs(const unsigned __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 219
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 220 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcs(const uchar2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 220
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 221 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcs(const uchar4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 221
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 222 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcs(const ushort2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 222
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcs(const ushort4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 223
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 224 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcs(const uint2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 224
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 225 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcs(const uint4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 225
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 226 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcs(const ulonglong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 226
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 228 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float __ldcs(const float *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 228
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 229 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double __ldcs(const double *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 229
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 230 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcs(const float2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 230
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 231 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcs(const float4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 231
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcs(const double2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 232
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 236 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline long __ldlu(const long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 236
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 237 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldlu(const unsigned long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 237
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 239 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char __ldlu(const char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 239
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 240 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldlu(const signed char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 240
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 241 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short __ldlu(const short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 241
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int __ldlu(const int *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 242
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 243 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldlu(const __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 243
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldlu(const char2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 244
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 245 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldlu(const char4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 245
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 246 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldlu(const short2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 246
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldlu(const short4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 247
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldlu(const int2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 248
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 249 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldlu(const int4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 249
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 250 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldlu(const longlong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 250
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldlu(const unsigned char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 252
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 253 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldlu(const unsigned short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 253
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 254 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldlu(const unsigned *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 254
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 255 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldlu(const unsigned __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 255
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 256 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldlu(const uchar2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 256
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 257 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldlu(const uchar4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 257
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 258 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldlu(const ushort2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 258
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 259 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldlu(const ushort4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 259
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldlu(const uint2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 260
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 261 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldlu(const uint4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 261
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 262 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldlu(const ulonglong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 262
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 264 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float __ldlu(const float *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 264
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 265 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double __ldlu(const double *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 265
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 266 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldlu(const float2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 266
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldlu(const float4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 267
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 268 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldlu(const double2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 268
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 272 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline long __ldcv(const long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 272
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 273 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcv(const unsigned long *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 273
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 275 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char __ldcv(const char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 275
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcv(const signed char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 276
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 277 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short __ldcv(const short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 277
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 278 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int __ldcv(const int *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 278
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcv(const __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 279
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcv(const char2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 280
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 281 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcv(const char4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 281
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 282 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcv(const short2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 282
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 283 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcv(const short4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 283
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 284 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcv(const int2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 284
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcv(const int4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 285
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 286 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcv(const longlong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 286
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcv(const unsigned char *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 288
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 289 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcv(const unsigned short *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 289
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 290 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcv(const unsigned *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 290
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 291 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcv(const unsigned __int64 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 291
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 292 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcv(const uchar2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 292
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 293 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcv(const uchar4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 293
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 294 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcv(const ushort2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 294
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 295 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcv(const ushort4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 295
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 296 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcv(const uint2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 296
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 297 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcv(const uint4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 297
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 298 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcv(const ulonglong2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 298
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 300 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float __ldcv(const float *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 300
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 301 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double __ldcv(const double *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 301
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 302 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcv(const float2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 302
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 303 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcv(const float4 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 303
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 304 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcv(const double2 *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 304
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 308 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(long *ptr, long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 308
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned long *ptr, unsigned long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 309
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 311 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char *ptr, char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 311
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(signed char *ptr, signed char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 312
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 313 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short *ptr, short value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 313
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 314 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int *ptr, int value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 314
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(__int64 *ptr, __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 315
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 316 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char2 *ptr, char2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 316
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char4 *ptr, char4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 317
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 318 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short2 *ptr, short2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 318
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 319 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short4 *ptr, short4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 319
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 320 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int2 *ptr, int2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 320
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 321 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int4 *ptr, int4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 321
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 322 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(longlong2 *ptr, longlong2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 322
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned char *ptr, unsigned char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 324
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 325 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned short *ptr, unsigned short value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 325
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 326 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned *ptr, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 326
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned __int64 *ptr, unsigned __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 327
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 328 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uchar2 *ptr, uchar2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 328
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 329 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uchar4 *ptr, uchar4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 329
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ushort2 *ptr, ushort2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 330
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 331 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ushort4 *ptr, ushort4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 331
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 332 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uint2 *ptr, uint2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 332
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uint4 *ptr, uint4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 333
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 334 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ulonglong2 *ptr, ulonglong2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 334
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 336 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float *ptr, float value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 336
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 337 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(double *ptr, double value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 337
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 338 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float2 *ptr, float2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 338
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float4 *ptr, float4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 339
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(double2 *ptr, double2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 340
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 344 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(long *ptr, long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 344
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned long *ptr, unsigned long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 345
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 347 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char *ptr, char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 347
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(signed char *ptr, signed char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 348
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 349 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short *ptr, short value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 349
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int *ptr, int value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 350
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 351 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(__int64 *ptr, __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 351
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 352 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char2 *ptr, char2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 352
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 353 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char4 *ptr, char4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 353
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 354 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short2 *ptr, short2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 354
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 355 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short4 *ptr, short4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 355
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int2 *ptr, int2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 356
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int4 *ptr, int4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 357
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 358 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(longlong2 *ptr, longlong2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 358
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 360 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned char *ptr, unsigned char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 360
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 361 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned short *ptr, unsigned short value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 361
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 362 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned *ptr, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 362
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned __int64 *ptr, unsigned __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 363
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 364 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uchar2 *ptr, uchar2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 364
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uchar4 *ptr, uchar4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 365
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 366 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ushort2 *ptr, ushort2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 366
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ushort4 *ptr, ushort4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 367
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uint2 *ptr, uint2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 368
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uint4 *ptr, uint4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 369
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ulonglong2 *ptr, ulonglong2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 370
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 372 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float *ptr, float value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 372
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 373 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(double *ptr, double value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 373
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 374 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float2 *ptr, float2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 374
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float4 *ptr, float4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 375
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(double2 *ptr, double2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 376
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(long *ptr, long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 380
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned long *ptr, unsigned long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 381
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char *ptr, char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 383
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(signed char *ptr, signed char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 384
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 385 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short *ptr, short value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 385
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 386 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int *ptr, int value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 386
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(__int64 *ptr, __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 387
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char2 *ptr, char2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 388
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 389 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char4 *ptr, char4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 389
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short2 *ptr, short2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 390
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 391 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short4 *ptr, short4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 391
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int2 *ptr, int2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 392
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int4 *ptr, int4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 393
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(longlong2 *ptr, longlong2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 394
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned char *ptr, unsigned char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 396
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned short *ptr, unsigned short value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 397
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 398 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned *ptr, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 398
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 399 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned __int64 *ptr, unsigned __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 399
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uchar2 *ptr, uchar2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 400
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uchar4 *ptr, uchar4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 401
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 402 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ushort2 *ptr, ushort2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 402
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 403 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ushort4 *ptr, ushort4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 403
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 404 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uint2 *ptr, uint2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 404
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uint4 *ptr, uint4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 405
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 406 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ulonglong2 *ptr, ulonglong2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 406
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float *ptr, float value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 408
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 409 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(double *ptr, double value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 409
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float2 *ptr, float2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 410
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float4 *ptr, float4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 411
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 412 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(double2 *ptr, double2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 412
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(long *ptr, long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 416
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned long *ptr, unsigned long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 417
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 419 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char *ptr, char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 419
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 420 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(signed char *ptr, signed char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 420
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 421 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short *ptr, short value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 421
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 422 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int *ptr, int value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 422
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(__int64 *ptr, __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 423
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 424 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char2 *ptr, char2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 424
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 425 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char4 *ptr, char4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 425
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 426 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short2 *ptr, short2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 426
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 427 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short4 *ptr, short4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 427
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 428 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int2 *ptr, int2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 428
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int4 *ptr, int4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 429
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 430 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(longlong2 *ptr, longlong2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 430
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned char *ptr, unsigned char value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 432
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 433 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned short *ptr, unsigned short value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 433
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 434 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned *ptr, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 434
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 435 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned __int64 *ptr, unsigned __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 435
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 436 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uchar2 *ptr, uchar2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 436
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 437 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uchar4 *ptr, uchar4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 437
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ushort2 *ptr, ushort2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 438
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 439 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ushort4 *ptr, ushort4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 439
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uint2 *ptr, uint2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 440
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 441 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uint4 *ptr, uint4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 441
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ulonglong2 *ptr, ulonglong2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 442
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 444 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float *ptr, float value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 444
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 445 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(double *ptr, double value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 445
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 446 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float2 *ptr, float2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 446
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 447 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float4 *ptr, float4 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 447
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 448 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(double2 *ptr, double2 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 448
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 465 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 465
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 477 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 477
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 490 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 490
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 502 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 502
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(int srcA, int srcB, int c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 102
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 113
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(short2 srcA, char4 srcB, int c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 125
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 136
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 148 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(int srcA, int srcB, int c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 148
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 159
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(short2 srcA, char4 srcB, int c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 171
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 182 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 182
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 197 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(int srcA, int srcB, int c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 197
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 206 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 206
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(char4 srcA, char4 srcB, int c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 216
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 225 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 225
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 98
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, int value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 99
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 100
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, long value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 101
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 102
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, __int64 value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 103
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, float value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 104
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, double value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 105
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 107
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, int value, int *pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 108
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 109
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, long value, int *pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 110
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned __int64 value, int *pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 111
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, __int64 value, int *pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 112
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, float value, int *pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 113
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, double value, int *pred) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 114
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline void __nanosleep(unsigned ns) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
#line 116
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 118
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_add_sync(unsigned mask, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 97
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_min_sync(unsigned mask, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 98
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_max_sync(unsigned mask, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 99
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_add_sync(unsigned mask, int value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 101
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_min_sync(unsigned mask, int value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 102
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_max_sync(unsigned mask, int value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 103
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_and_sync(unsigned mask, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 105
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_or_sync(unsigned mask, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 106
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_xor_sync(unsigned mask, unsigned value) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
extern "C" {
#line 113
inline void *__nv_associate_access_property(const void *ptr, unsigned __int64 
#line 114
property) {int volatile ___ = 1;(void)ptr;(void)property;::exit(___);}
#if 0
#line 114
{ 
#line 115
extern void *__nv_associate_access_property_impl(const void *, unsigned __int64); 
#line 117
return __nv_associate_access_property_impl(ptr, property); 
#line 118
} 
#endif
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
inline void __nv_memcpy_async_shared_global_4(void *dst, const void *
#line 121
src, unsigned 
#line 122
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
#line 122
{ 
#line 123
extern void __nv_memcpy_async_shared_global_4_impl(void *, const void *, unsigned); 
#line 126
__nv_memcpy_async_shared_global_4_impl(dst, src, src_size); 
#line 127
} 
#endif
#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
inline void __nv_memcpy_async_shared_global_8(void *dst, const void *
#line 130
src, unsigned 
#line 131
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
#line 131
{ 
#line 132
extern void __nv_memcpy_async_shared_global_8_impl(void *, const void *, unsigned); 
#line 135
__nv_memcpy_async_shared_global_8_impl(dst, src, src_size); 
#line 136
} 
#endif
#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
inline void __nv_memcpy_async_shared_global_16(void *dst, const void *
#line 139
src, unsigned 
#line 140
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
#line 140
{ 
#line 141
extern void __nv_memcpy_async_shared_global_16_impl(void *, const void *, unsigned); 
#line 144
__nv_memcpy_async_shared_global_16_impl(dst, src, src_size); 
#line 145
} 
#endif
#line 147 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_80_rt.h"
}
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 92 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __isCtaShared(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 92
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __isClusterShared(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 93
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline void *__cluster_map_shared_rank(const void *ptr, unsigned target_block_rank) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)target_block_rank;::exit(___);}
#if 0
#line 94
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __cluster_query_shared_rank(const void *ptr) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 95
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline uint2 __cluster_map_shared_multicast(const void *ptr, unsigned cluster_cta_mask) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)cluster_cta_mask;::exit(___);}
#if 0
#line 96
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __clusterDimIsSpecified() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 97
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline dim3 __clusterDim() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 98
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline dim3 __clusterRelativeBlockIdx() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 99
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline dim3 __clusterGridDimInClusters() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 100
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline dim3 __clusterIdx() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 101
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __clusterRelativeBlockRank() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 102
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __clusterSizeInBlocks() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 103
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline void __cluster_barrier_arrive() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 104
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline void __cluster_barrier_arrive_relaxed() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 105
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline void __cluster_barrier_wait() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 106
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline void __threadfence_cluster() 
__pragma(warning(pop))
{int volatile ___ = 1;::exit(___);}
#if 0
#line 107
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline float2 atomicAdd(float2 *__address, float2 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
#line 109
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline float2 atomicAdd_block(float2 *__address, float2 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
#line 110
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline float2 atomicAdd_system(float2 *__address, float2 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
#line 111
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline float4 atomicAdd(float4 *__address, float4 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
#line 112
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline float4 atomicAdd_block(float4 *__address, float4 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
#line 113
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static __inline float4 atomicAdd_system(float4 *__address, float4 val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)val;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
extern "C" {
#line 132
}
#line 139
template< bool __b, class _T> 
#line 140
struct __nv_atomic_enable_if { }; 
#line 142
template< class _T> 
#line 143
struct __nv_atomic_enable_if< true, _T>  { typedef _T __type; }; 
#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
template< class _T> 
#line 154
struct __nv_atomic_triv_cp_helper { 
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
static const bool __val = __is_trivially_copyable(_T); 
#line 166 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
}; 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 201 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
template< class _T> static __inline typename __nv_atomic_enable_if< ((sizeof(_T) == (16)) && (__alignof(_T) >= (16))) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
#line 203
atomicCAS(_T *__address, _T __compare, _T __val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)__compare;(void)__val;::exit(___);}
#if 0
#line 203
{ 
#line 204
union _U { _T __ret; __inline _U() {int *volatile ___ = 0;::free(___);}
#if 0
#line 204
{ } 
#endif
#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
}; _U __u; 
#line 205
__u128AtomicCAS((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__compare)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
#line 209
return __u.__ret; 
#line 210
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
template< class _T> static __inline typename __nv_atomic_enable_if< ((sizeof(_T) == (16)) && (__alignof(_T) >= (16))) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
#line 214
atomicCAS_block(_T *__address, _T __compare, _T __val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)__compare;(void)__val;::exit(___);}
#if 0
#line 214
{ 
#line 215
union _U { _T __ret; __inline _U() {int *volatile ___ = 0;::free(___);}
#if 0
#line 215
{ } 
#endif
#line 215 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
}; _U __u; 
#line 216
__u128AtomicCAS_block((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__compare)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
#line 220
return __u.__ret; 
#line 221
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
template< class _T> static __inline typename __nv_atomic_enable_if< ((sizeof(_T) == (16)) && (__alignof(_T) >= (16))) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
#line 225
atomicCAS_system(_T *__address, _T __compare, _T __val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)__compare;(void)__val;::exit(___);}
#if 0
#line 225
{ 
#line 226
union _U { _T __ret; __inline _U() {int *volatile ___ = 0;::free(___);}
#if 0
#line 226
{ } 
#endif
#line 226 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
}; _U __u; 
#line 227
__u128AtomicCAS_system((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__compare)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
#line 231
return __u.__ret; 
#line 232
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
template< class _T> static __inline typename __nv_atomic_enable_if< ((sizeof(_T) == (16)) && (__alignof(_T) >= (16))) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
#line 236
atomicExch(_T *__address, _T __val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)__val;::exit(___);}
#if 0
#line 236
{ 
#line 237
union _U { _T __ret; __inline _U() {int *volatile ___ = 0;::free(___);}
#if 0
#line 237
{ } 
#endif
#line 237 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
}; _U __u; 
#line 238
__u128AtomicExch((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
#line 241
return __u.__ret; 
#line 242
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
template< class _T> static __inline typename __nv_atomic_enable_if< ((sizeof(_T) == (16)) && (__alignof(_T) >= (16))) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
#line 246
atomicExch_block(_T *__address, _T __val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)__val;::exit(___);}
#if 0
#line 246
{ 
#line 247
union _U { _T __ret; __inline _U() {int *volatile ___ = 0;::free(___);}
#if 0
#line 247
{ } 
#endif
#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
}; _U __u; 
#line 248
__u128AtomicExch_block((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
#line 251
return __u.__ret; 
#line 252
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 254 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
template< class _T> static __inline typename __nv_atomic_enable_if< ((sizeof(_T) == (16)) && (__alignof(_T) >= (16))) && __nv_atomic_triv_cp_helper< _T> ::__val, _T> ::__type 
#line 256
atomicExch_system(_T *__address, _T __val) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)__address;(void)__val;::exit(___);}
#if 0
#line 256
{ 
#line 257
union _U { _T __ret; __inline _U() {int *volatile ___ = 0;::free(___);}
#if 0
#line 257
{ } 
#endif
#line 257 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_90_rt.h"
}; _U __u; 
#line 258
__u128AtomicExch_system((void *)__address, (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__val)))), (void *)(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__u.__ret))))); 
#line 261
return __u.__ret; 
#line 262
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __ffma2_rn(float2 x, float2 y, float2 z) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 100
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __ffma2_rz(float2 x, float2 y, float2 z) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 112
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __ffma2_rd(float2 x, float2 y, float2 z) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 124
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __ffma2_ru(float2 x, float2 y, float2 z) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 136
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __fadd2_rn(float2 x, float2 y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;::exit(___);}
#if 0
#line 149
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 161 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __fadd2_rz(float2 x, float2 y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;::exit(___);}
#if 0
#line 161
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __fadd2_rd(float2 x, float2 y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;::exit(___);}
#if 0
#line 173
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __fadd2_ru(float2 x, float2 y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;::exit(___);}
#if 0
#line 185
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 198 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __fmul2_rn(float2 x, float2 y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;::exit(___);}
#if 0
#line 198
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 210 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __fmul2_rz(float2 x, float2 y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;::exit(___);}
#if 0
#line 210
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 222 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __fmul2_rd(float2 x, float2 y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;::exit(___);}
#if 0
#line 222
{ } 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt\\sm_100_rt.h"
static __inline float2 __fmul2_ru(float2 x, float2 y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)x;(void)y;::exit(___);}
#if 0
#line 234
{ } 
#endif
#line 65 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
#line 66
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
#line 67
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
#line 68
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
#line 69
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
#line 70
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
#line 71
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
#line 72
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
#line 73
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
#line 74
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
#line 75
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
#line 76
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
#line 77
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
#line 78
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
#line 79
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
#line 80
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
#line 81
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
#line 82
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
#line 83
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
#line 84
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
#line 85
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
#line 86
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
#line 87
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
#line 88
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
#line 89
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
#line 90
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
#line 92
template<> struct __nv_itex_trait< long>  { typedef void type; }; 
#line 93
template<> struct __nv_itex_trait< long1>  { typedef void type; }; 
#line 94
template<> struct __nv_itex_trait< long2>  { typedef void type; }; 
#line 95
template<> struct __nv_itex_trait< long4>  { typedef void type; }; 
#line 96
template<> struct __nv_itex_trait< unsigned long>  { typedef void type; }; 
#line 97
template<> struct __nv_itex_trait< ulong1>  { typedef void type; }; 
#line 98
template<> struct __nv_itex_trait< ulong2>  { typedef void type; }; 
#line 99
template<> struct __nv_itex_trait< ulong4>  { typedef void type; }; 
#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
#line 102
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
#line 103
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
#line 104
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 108
template< class T> static typename __nv_itex_trait< T> ::type 
#line 109
tex1Dfetch(T *ptr, ::cudaTextureObject_t obj, int x) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;::exit(___);}
#if 0
#line 110
{ 
#line 111
__nv_tex_surf_handler("__itex1Dfetch", ptr, obj, x); 
#line 112
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 115
tex1Dfetch(::cudaTextureObject_t texObject, int x) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;::exit(___);}
#if 0
#line 116
{ 
#line 117
T ret; 
#line 118
tex1Dfetch(&ret, texObject, x); 
#line 119
return ret; 
#line 120
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 123
tex1D(T *ptr, ::cudaTextureObject_t obj, float x) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;::exit(___);}
#if 0
#line 124
{ 
#line 125
__nv_tex_surf_handler("__itex1D", ptr, obj, x); 
#line 126
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 130
tex1D(::cudaTextureObject_t texObject, float x) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;::exit(___);}
#if 0
#line 131
{ 
#line 132
T ret; 
#line 133
tex1D(&ret, texObject, x); 
#line 134
return ret; 
#line 135
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 139
tex2D(T *ptr, ::cudaTextureObject_t obj, float x, float y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;::exit(___);}
#if 0
#line 140
{ 
#line 141
__nv_tex_surf_handler("__itex2D", ptr, obj, x, y); 
#line 142
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 145
tex2D(::cudaTextureObject_t texObject, float x, float y) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;::exit(___);}
#if 0
#line 146
{ 
#line 147
T ret; 
#line 148
tex2D(&ret, texObject, x, y); 
#line 149
return ret; 
#line 150
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 154
tex2D(T *ptr, ::cudaTextureObject_t obj, float x, float y, bool *
#line 155
isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;::exit(___);}
#if 0
#line 156
{ 
#line 157
unsigned char res; 
#line 158
__nv_tex_surf_handler("__itex2D_sparse", ptr, obj, x, y, &res); 
#line 159
(*isResident) = (res != 0); 
#line 160
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 163
tex2D(::cudaTextureObject_t texObject, float x, float y, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)isResident;::exit(___);}
#if 0
#line 164
{ 
#line 165
T ret; 
#line 166
tex2D(&ret, texObject, x, y, isResident); 
#line 167
return ret; 
#line 168
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 174
tex3D(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 175
{ 
#line 176
__nv_tex_surf_handler("__itex3D", ptr, obj, x, y, z); 
#line 177
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 180
tex3D(::cudaTextureObject_t texObject, float x, float y, float z) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 181
{ 
#line 182
T ret; 
#line 183
tex3D(&ret, texObject, x, y, z); 
#line 184
return ret; 
#line 185
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 189
tex3D(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, bool *
#line 190
isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)isResident;::exit(___);}
#if 0
#line 191
{ 
#line 192
unsigned char res; 
#line 193
__nv_tex_surf_handler("__itex3D_sparse", ptr, obj, x, y, z, &res); 
#line 194
(*isResident) = (res != 0); 
#line 195
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 197 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 198
tex3D(::cudaTextureObject_t texObject, float x, float y, float z, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)isResident;::exit(___);}
#if 0
#line 199
{ 
#line 200
T ret; 
#line 201
tex3D(&ret, texObject, x, y, z, isResident); 
#line 202
return ret; 
#line 203
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 208
tex1DLayered(T *ptr, ::cudaTextureObject_t obj, float x, int layer) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;::exit(___);}
#if 0
#line 209
{ 
#line 210
__nv_tex_surf_handler("__itex1DLayered", ptr, obj, x, layer); 
#line 211
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 214
tex1DLayered(::cudaTextureObject_t texObject, float x, int layer) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;::exit(___);}
#if 0
#line 215
{ 
#line 216
T ret; 
#line 217
tex1DLayered(&ret, texObject, x, layer); 
#line 218
return ret; 
#line 219
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 221 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 222
tex2DLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 223
{ 
#line 224
__nv_tex_surf_handler("__itex2DLayered", ptr, obj, x, y, layer); 
#line 225
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 227 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 228
tex2DLayered(::cudaTextureObject_t texObject, float x, float y, int layer) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 229
{ 
#line 230
T ret; 
#line 231
tex2DLayered(&ret, texObject, x, y, layer); 
#line 232
return ret; 
#line 233
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 236 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 237
tex2DLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)isResident;::exit(___);}
#if 0
#line 238
{ 
#line 239
unsigned char res; 
#line 240
__nv_tex_surf_handler("__itex2DLayered_sparse", ptr, obj, x, y, layer, &res); 
#line 241
(*isResident) = (res != 0); 
#line 242
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 245
tex2DLayered(::cudaTextureObject_t texObject, float x, float y, int layer, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)isResident;::exit(___);}
#if 0
#line 246
{ 
#line 247
T ret; 
#line 248
tex2DLayered(&ret, texObject, x, y, layer, isResident); 
#line 249
return ret; 
#line 250
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 254 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 255
texCubemap(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 256
{ 
#line 257
__nv_tex_surf_handler("__itexCubemap", ptr, obj, x, y, z); 
#line 258
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 261 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 262
texCubemap(::cudaTextureObject_t texObject, float x, float y, float z) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 263
{ 
#line 264
T ret; 
#line 265
texCubemap(&ret, texObject, x, y, z); 
#line 266
return ret; 
#line 267
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 270 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 271
texCubemapLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 272
{ 
#line 273
__nv_tex_surf_handler("__itexCubemapLayered", ptr, obj, x, y, z, layer); 
#line 274
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 277
texCubemapLayered(::cudaTextureObject_t texObject, float x, float y, float z, int layer) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 278
{ 
#line 279
T ret; 
#line 280
texCubemapLayered(&ret, texObject, x, y, z, layer); 
#line 281
return ret; 
#line 282
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 284 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 285
tex2Dgather(T *ptr, ::cudaTextureObject_t obj, float x, float y, int comp = 0) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 286
{ 
#line 287
__nv_tex_surf_handler("__itex2Dgather", ptr, obj, x, y, comp); 
#line 288
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 290 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 291
tex2Dgather(::cudaTextureObject_t to, float x, float y, int comp = 0) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 292
{ 
#line 293
T ret; 
#line 294
tex2Dgather(&ret, to, x, y, comp); 
#line 295
return ret; 
#line 296
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 300
tex2Dgather(T *ptr, ::cudaTextureObject_t obj, float x, float y, bool *isResident, int comp = 0) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;(void)comp;::exit(___);}
#if 0
#line 301
{ 
#line 302
unsigned char res; 
#line 303
__nv_tex_surf_handler("__itex2Dgather_sparse", ptr, obj, x, y, comp, &res); 
#line 304
(*isResident) = (res != 0); 
#line 305
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 307 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 308
tex2Dgather(::cudaTextureObject_t to, float x, float y, bool *isResident, int comp = 0) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)isResident;(void)comp;::exit(___);}
#if 0
#line 309
{ 
#line 310
T ret; 
#line 311
tex2Dgather(&ret, to, x, y, isResident, comp); 
#line 312
return ret; 
#line 313
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 318
tex1DLod(T *ptr, ::cudaTextureObject_t obj, float x, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;::exit(___);}
#if 0
#line 319
{ 
#line 320
__nv_tex_surf_handler("__itex1DLod", ptr, obj, x, level); 
#line 321
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 323 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 324
tex1DLod(::cudaTextureObject_t texObject, float x, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;::exit(___);}
#if 0
#line 325
{ 
#line 326
T ret; 
#line 327
tex1DLod(&ret, texObject, x, level); 
#line 328
return ret; 
#line 329
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 332 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 333
tex2DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 334
{ 
#line 335
__nv_tex_surf_handler("__itex2DLod", ptr, obj, x, y, level); 
#line 336
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 338 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 339
tex2DLod(::cudaTextureObject_t texObject, float x, float y, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 340
{ 
#line 341
T ret; 
#line 342
tex2DLod(&ret, texObject, x, y, level); 
#line 343
return ret; 
#line 344
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 349
tex2DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float level, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;(void)isResident;::exit(___);}
#if 0
#line 350
{ 
#line 351
unsigned char res; 
#line 352
__nv_tex_surf_handler("__itex2DLod_sparse", ptr, obj, x, y, level, &res); 
#line 353
(*isResident) = (res != 0); 
#line 354
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 357
tex2DLod(::cudaTextureObject_t texObject, float x, float y, float level, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;(void)isResident;::exit(___);}
#if 0
#line 358
{ 
#line 359
T ret; 
#line 360
tex2DLod(&ret, texObject, x, y, level, isResident); 
#line 361
return ret; 
#line 362
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 368
tex3DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 369
{ 
#line 370
__nv_tex_surf_handler("__itex3DLod", ptr, obj, x, y, z, level); 
#line 371
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 373 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 374
tex3DLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 375
{ 
#line 376
T ret; 
#line 377
tex3DLod(&ret, texObject, x, y, z, level); 
#line 378
return ret; 
#line 379
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 383
tex3DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;(void)isResident;::exit(___);}
#if 0
#line 384
{ 
#line 385
unsigned char res; 
#line 386
__nv_tex_surf_handler("__itex3DLod_sparse", ptr, obj, x, y, z, level, &res); 
#line 387
(*isResident) = (res != 0); 
#line 388
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 391
tex3DLod(::cudaTextureObject_t texObject, float x, float y, float z, float level, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;(void)isResident;::exit(___);}
#if 0
#line 392
{ 
#line 393
T ret; 
#line 394
tex3DLod(&ret, texObject, x, y, z, level, isResident); 
#line 395
return ret; 
#line 396
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 402
tex1DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, int layer, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 403
{ 
#line 404
__nv_tex_surf_handler("__itex1DLayeredLod", ptr, obj, x, layer, level); 
#line 405
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 407 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 408
tex1DLayeredLod(::cudaTextureObject_t texObject, float x, int layer, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 409
{ 
#line 410
T ret; 
#line 411
tex1DLayeredLod(&ret, texObject, x, layer, level); 
#line 412
return ret; 
#line 413
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 417
tex2DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 418
{ 
#line 419
__nv_tex_surf_handler("__itex2DLayeredLod", ptr, obj, x, y, layer, level); 
#line 420
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 422 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 423
tex2DLayeredLod(::cudaTextureObject_t texObject, float x, float y, int layer, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 424
{ 
#line 425
T ret; 
#line 426
tex2DLayeredLod(&ret, texObject, x, y, layer, level); 
#line 427
return ret; 
#line 428
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 431 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 432
tex2DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, float level, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;(void)isResident;::exit(___);}
#if 0
#line 433
{ 
#line 434
unsigned char res; 
#line 435
__nv_tex_surf_handler("__itex2DLayeredLod_sparse", ptr, obj, x, y, layer, level, &res); 
#line 436
(*isResident) = (res != 0); 
#line 437
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 439 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 440
tex2DLayeredLod(::cudaTextureObject_t texObject, float x, float y, int layer, float level, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;(void)isResident;::exit(___);}
#if 0
#line 441
{ 
#line 442
T ret; 
#line 443
tex2DLayeredLod(&ret, texObject, x, y, layer, level, isResident); 
#line 444
return ret; 
#line 445
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 448 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 449
texCubemapLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 450
{ 
#line 451
__nv_tex_surf_handler("__itexCubemapLod", ptr, obj, x, y, z, level); 
#line 452
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 454 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 455
texCubemapLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 456
{ 
#line 457
T ret; 
#line 458
texCubemapLod(&ret, texObject, x, y, z, level); 
#line 459
return ret; 
#line 460
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 463 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 464
texCubemapGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 465
{ 
#line 466
__nv_tex_surf_handler("__itexCubemapGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy); 
#line 467
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 469 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 470
texCubemapGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 471
{ 
#line 472
T ret; 
#line 473
texCubemapGrad(&ret, texObject, x, y, z, dPdx, dPdy); 
#line 474
return ret; 
#line 475
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 477 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 478
texCubemapLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 479
{ 
#line 480
__nv_tex_surf_handler("__itexCubemapLayeredLod", ptr, obj, x, y, z, layer, level); 
#line 481
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 483 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 484
texCubemapLayeredLod(::cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 485
{ 
#line 486
T ret; 
#line 487
texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level); 
#line 488
return ret; 
#line 489
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 491 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 492
tex1DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 493
{ 
#line 494
__nv_tex_surf_handler("__itex1DGrad", ptr, obj, x, dPdx, dPdy); 
#line 495
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 497 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 498
tex1DGrad(::cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 499
{ 
#line 500
T ret; 
#line 501
tex1DGrad(&ret, texObject, x, dPdx, dPdy); 
#line 502
return ret; 
#line 503
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 506 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 507
tex2DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, ::float2 dPdx, ::float2 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 508
{ 
#line 509
__nv_tex_surf_handler("__itex2DGrad_v2", ptr, obj, x, y, &dPdx, &dPdy); 
#line 510
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 512 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 513
tex2DGrad(::cudaTextureObject_t texObject, float x, float y, ::float2 dPdx, ::float2 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 514
{ 
#line 515
T ret; 
#line 516
tex2DGrad(&ret, texObject, x, y, dPdx, dPdy); 
#line 517
return ret; 
#line 518
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 521 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 522
tex2DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, ::float2 dPdx, ::float2 dPdy, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 523
{ 
#line 524
unsigned char res; 
#line 525
__nv_tex_surf_handler("__itex2DGrad_sparse", ptr, obj, x, y, &dPdx, &dPdy, &res); 
#line 526
(*isResident) = (res != 0); 
#line 527
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 530
tex2DGrad(::cudaTextureObject_t texObject, float x, float y, ::float2 dPdx, ::float2 dPdy, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 531
{ 
#line 532
T ret; 
#line 533
tex2DGrad(&ret, texObject, x, y, dPdx, dPdy, isResident); 
#line 534
return ret; 
#line 535
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 539 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 540
tex3DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 541
{ 
#line 542
__nv_tex_surf_handler("__itex3DGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy); 
#line 543
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 545 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 546
tex3DGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 547
{ 
#line 548
T ret; 
#line 549
tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy); 
#line 550
return ret; 
#line 551
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 554 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 555
tex3DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 556
{ 
#line 557
unsigned char res; 
#line 558
__nv_tex_surf_handler("__itex3DGrad_sparse", ptr, obj, x, y, z, &dPdx, &dPdy, &res); 
#line 559
(*isResident) = (res != 0); 
#line 560
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 562 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 563
tex3DGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 564
{ 
#line 565
T ret; 
#line 566
tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy, isResident); 
#line 567
return ret; 
#line 568
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 573 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 574
tex1DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 575
{ 
#line 576
__nv_tex_surf_handler("__itex1DLayeredGrad", ptr, obj, x, layer, dPdx, dPdy); 
#line 577
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 579 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 580
tex1DLayeredGrad(::cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 581
{ 
#line 582
T ret; 
#line 583
tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy); 
#line 584
return ret; 
#line 585
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 588 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 589
tex2DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 590
{ 
#line 591
__nv_tex_surf_handler("__itex2DLayeredGrad_v2", ptr, obj, x, y, layer, &dPdx, &dPdy); 
#line 592
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 594 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 595
tex2DLayeredGrad(::cudaTextureObject_t texObject, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 596
{ 
#line 597
T ret; 
#line 598
tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy); 
#line 599
return ret; 
#line 600
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 603 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 604
tex2DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 605
{ 
#line 606
unsigned char res; 
#line 607
__nv_tex_surf_handler("__itex2DLayeredGrad_sparse", ptr, obj, x, y, layer, &dPdx, &dPdy, &res); 
#line 608
(*isResident) = (res != 0); 
#line 609
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 611 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 612
tex2DLayeredGrad(::cudaTextureObject_t texObject, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy, bool *isResident) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 613
{ 
#line 614
T ret; 
#line 615
tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy, isResident); 
#line 616
return ret; 
#line 617
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 621 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 622
texCubemapLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 623
{ 
#line 624
__nv_tex_surf_handler("__itexCubemapLayeredGrad_v2", ptr, obj, x, y, z, layer, &dPdx, &dPdy); 
#line 625
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 627 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 628
texCubemapLayeredGrad(::cudaTextureObject_t texObject, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 629
{ 
#line 630
T ret; 
#line 631
texCubemapLayeredGrad(&ret, texObject, x, y, z, layer, dPdx, dPdy); 
#line 632
return ret; 
#line 633
} 
#endif
#line 58 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
#line 59
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
#line 60
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
#line 61
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
#line 62
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
#line 63
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
#line 64
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
#line 65
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
#line 66
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
#line 67
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
#line 68
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
#line 69
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
#line 70
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
#line 71
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
#line 72
template<> struct __nv_isurf_trait< __int64>  { typedef void type; }; 
#line 73
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
#line 74
template<> struct __nv_isurf_trait< unsigned __int64>  { typedef void type; }; 
#line 75
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
#line 76
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
#line 77
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
#line 79
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
#line 80
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
#line 81
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
#line 82
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
#line 83
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
#line 84
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
#line 85
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
#line 86
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
#line 87
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
#line 89
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
#line 90
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
#line 91
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
#line 92
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
#line 93
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
#line 94
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
#line 95
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 98
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 99
surf1Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;::exit(___);}
#if 0
#line 100
{ 
#line 101
__nv_tex_surf_handler("__isurf1Dread", ptr, obj, x, mode); 
#line 102
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 105
surf1Dread(::cudaSurfaceObject_t surfObject, int x, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;::exit(___);}
#if 0
#line 106
{ 
#line 107
T ret; 
#line 108
surf1Dread(&ret, surfObject, x, boundaryMode); 
#line 109
return ret; 
#line 110
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 113
surf2Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 114
{ 
#line 115
__nv_tex_surf_handler("__isurf2Dread", ptr, obj, x, y, mode); 
#line 116
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 119
surf2Dread(::cudaSurfaceObject_t surfObject, int x, int y, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;::exit(___);}
#if 0
#line 120
{ 
#line 121
T ret; 
#line 122
surf2Dread(&ret, surfObject, x, y, boundaryMode); 
#line 123
return ret; 
#line 124
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 128
surf3Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 129
{ 
#line 130
__nv_tex_surf_handler("__isurf3Dread", ptr, obj, x, y, z, mode); 
#line 131
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 134
surf3Dread(::cudaSurfaceObject_t surfObject, int x, int y, int z, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;::exit(___);}
#if 0
#line 135
{ 
#line 136
T ret; 
#line 137
surf3Dread(&ret, surfObject, x, y, z, boundaryMode); 
#line 138
return ret; 
#line 139
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 142
surf1DLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 143
{ 
#line 144
__nv_tex_surf_handler("__isurf1DLayeredread", ptr, obj, x, layer, mode); 
#line 145
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 147 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 148
surf1DLayeredread(::cudaSurfaceObject_t surfObject, int x, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;::exit(___);}
#if 0
#line 149
{ 
#line 150
T ret; 
#line 151
surf1DLayeredread(&ret, surfObject, x, layer, boundaryMode); 
#line 152
return ret; 
#line 153
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 155 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 156
surf2DLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 157
{ 
#line 158
__nv_tex_surf_handler("__isurf2DLayeredread", ptr, obj, x, y, layer, mode); 
#line 159
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 161 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 162
surf2DLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;::exit(___);}
#if 0
#line 163
{ 
#line 164
T ret; 
#line 165
surf2DLayeredread(&ret, surfObject, x, y, layer, boundaryMode); 
#line 166
return ret; 
#line 167
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 170
surfCubemapread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 171
{ 
#line 172
__nv_tex_surf_handler("__isurfCubemapread", ptr, obj, x, y, face, mode); 
#line 173
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 176
surfCubemapread(::cudaSurfaceObject_t surfObject, int x, int y, int face, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;::exit(___);}
#if 0
#line 177
{ 
#line 178
T ret; 
#line 179
surfCubemapread(&ret, surfObject, x, y, face, boundaryMode); 
#line 180
return ret; 
#line 181
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 184
surfCubemapLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int layerface, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;::exit(___);}
#if 0
#line 185
{ 
#line 186
__nv_tex_surf_handler("__isurfCubemapLayeredread", ptr, obj, x, y, layerface, mode); 
#line 187
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 190
surfCubemapLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layerface, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;::exit(___);}
#if 0
#line 191
{ 
#line 192
T ret; 
#line 193
surfCubemapLayeredread(&ret, surfObject, x, y, layerface, boundaryMode); 
#line 194
return ret; 
#line 195
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 197 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 198
surf1Dwrite(T val, ::cudaSurfaceObject_t obj, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;::exit(___);}
#if 0
#line 199
{ 
#line 200
__nv_tex_surf_handler("__isurf1Dwrite_v2", &val, obj, x, mode); 
#line 201
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 203 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 204
surf2Dwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 205
{ 
#line 206
__nv_tex_surf_handler("__isurf2Dwrite_v2", &val, obj, x, y, mode); 
#line 207
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 209 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 210
surf3Dwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 211
{ 
#line 212
__nv_tex_surf_handler("__isurf3Dwrite_v2", &val, obj, x, y, z, mode); 
#line 213
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 215 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 216
surf1DLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 217
{ 
#line 218
__nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, obj, x, layer, mode); 
#line 219
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 221 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 222
surf2DLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 223
{ 
#line 224
__nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, obj, x, y, layer, mode); 
#line 225
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 227 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 228
surfCubemapwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 229
{ 
#line 230
__nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, obj, x, y, face, mode); 
#line 231
} 
#endif

__pragma(warning(push))
__pragma(warning(disable : 4505)) /* "unreferenced local function has been removed" */

#line 233 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 234
surfCubemapLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int layerface, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
__pragma(warning(pop))
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;::exit(___);}
#if 0
#line 235
{ 
#line 236
__nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, obj, x, y, layerface, mode); 
#line 237
} 
#endif
#line 2924 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt/device_functions.h"
extern "C" unsigned __stdcall __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
#line 2969 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\crt/device_functions.h"
enum { 
#line 2970
__NV_ATOMIC_RELAXED, 
#line 2971
__NV_ATOMIC_CONSUME, 
#line 2972
__NV_ATOMIC_ACQUIRE, 
#line 2973
__NV_ATOMIC_RELEASE, 
#line 2974
__NV_ATOMIC_ACQ_REL, 
#line 2975
__NV_ATOMIC_SEQ_CST
#line 2976
}; 
#line 2978
enum { 
#line 2979
__NV_THREAD_SCOPE_THREAD, 
#line 2980
__NV_THREAD_SCOPE_BLOCK, 
#line 2981
__NV_THREAD_SCOPE_CLUSTER, 
#line 2982
__NV_THREAD_SCOPE_DEVICE, 
#line 2983
__NV_THREAD_SCOPE_SYSTEM
#line 2984
}; 
#line 68 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_launch_parameters.h"
extern "C" {
#line 71 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\device_launch_parameters.h"
extern const uint3 __device_builtin_variable_threadIdx; 
#line 72
extern const uint3 __device_builtin_variable_blockIdx; 
#line 73
extern const dim3 __device_builtin_variable_blockDim; 
#line 74
extern const dim3 __device_builtin_variable_gridDim; 
#line 75
extern const int __device_builtin_variable_warpSize; 
#line 80
}
#line 14 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cstddef"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 27
namespace std { 
#line 28
using ::ptrdiff_t;
#line 30
using max_align_t = double; 
#line 31
using nullptr_t = decltype(nullptr); 
#line 99
}
#line 102
using std::max_align_t;
#line 111
#pragma warning(pop)
#pragma pack ( pop )
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\initializer_list"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 19
namespace std { 
#line 20
template < class _Elem >
class initializer_list {
public :
    using value_type = _Elem;
    using reference = const _Elem &;
    using const_reference = const _Elem &;
    using size_type = size_t;

    using iterator = const _Elem *;
    using const_iterator = const _Elem *;

    constexpr initializer_list ( ) noexcept : _First ( nullptr ), _Last ( nullptr ) { }

    constexpr initializer_list ( const _Elem * _First_arg, const _Elem * _Last_arg ) noexcept
        : _First ( _First_arg ), _Last ( _Last_arg ) { }

    [ [ nodiscard ] ] constexpr const _Elem * begin ( ) const noexcept {
        return _First;
    }

    [ [ nodiscard ] ] constexpr const _Elem * end ( ) const noexcept {
        return _Last;
    }

    [ [ nodiscard ] ] constexpr size_t size ( ) const noexcept {
        return static_cast < size_t > ( _Last - _First );
    }

private :
    const _Elem * _First;
    const _Elem * _Last;
};
#line 53
template < class _Elem >
[ [ nodiscard ] ] constexpr const _Elem * begin ( initializer_list < _Elem > _Ilist ) noexcept {
    return _Ilist . begin ( );
}
#line 58
template < class _Elem >
[ [ nodiscard ] ] constexpr const _Elem * end ( initializer_list < _Elem > _Ilist ) noexcept {
    return _Ilist . end ( );
}
#line 62
}
#line 65
#pragma warning(pop)
#pragma pack ( pop )
#line 15 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\stdint.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 18
typedef signed char int8_t; 
#line 19
typedef short int16_t; 
#line 20
typedef int int32_t; 
#line 21
typedef __int64 int64_t; 
#line 22
typedef unsigned char uint8_t; 
#line 23
typedef unsigned short uint16_t; 
#line 24
typedef unsigned uint32_t; 
#line 25
typedef unsigned __int64 uint64_t; 
#line 27
typedef signed char int_least8_t; 
#line 28
typedef short int_least16_t; 
#line 29
typedef int int_least32_t; 
#line 30
typedef __int64 int_least64_t; 
#line 31
typedef unsigned char uint_least8_t; 
#line 32
typedef unsigned short uint_least16_t; 
#line 33
typedef unsigned uint_least32_t; 
#line 34
typedef unsigned __int64 uint_least64_t; 
#line 36
typedef signed char int_fast8_t; 
#line 37
typedef int int_fast16_t; 
#line 38
typedef int int_fast32_t; 
#line 39
typedef __int64 int_fast64_t; 
#line 40
typedef unsigned char uint_fast8_t; 
#line 41
typedef unsigned uint_fast16_t; 
#line 42
typedef unsigned uint_fast32_t; 
#line 43
typedef unsigned __int64 uint_fast64_t; 
#line 45
typedef __int64 intmax_t; 
#line 46
typedef unsigned __int64 uintmax_t; 
#line 136 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\stdint.h"
#pragma warning(pop)
#line 13 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cstdint"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 20
namespace std { 
#line 21
using ::int8_t;
#line 22
using ::int16_t;
#line 23
using ::int32_t;
#line 24
using ::int64_t;
#line 25
using ::uint8_t;
#line 26
using ::uint16_t;
#line 27
using ::uint32_t;
#line 28
using ::uint64_t;
#line 30
using ::int_least8_t;
#line 31
using ::int_least16_t;
#line 32
using ::int_least32_t;
#line 33
using ::int_least64_t;
#line 34
using ::uint_least8_t;
#line 35
using ::uint_least16_t;
#line 36
using ::uint_least32_t;
#line 37
using ::uint_least64_t;
#line 39
using ::int_fast8_t;
#line 40
using ::int_fast16_t;
#line 41
using ::int_fast32_t;
#line 42
using ::int_fast64_t;
#line 43
using ::uint_fast8_t;
#line 44
using ::uint_fast16_t;
#line 45
using ::uint_fast32_t;
#line 46
using ::uint_fast64_t;
#line 48
using ::intmax_t;
#line 49
using ::intptr_t;
#line 50
using ::uintmax_t;
#line 51
using ::uintptr_t;
#line 54
namespace [[deprecated("warning STL4002: The non-Standard std::tr1 namespace and TR1-only machinery are deprecated and will be REMOVED. You can define _" "SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING to suppress this warning.")]] tr1 { 
#line 55
using ::int8_t;
#line 56
using ::int16_t;
#line 57
using ::int32_t;
#line 58
using ::int64_t;
#line 59
using ::uint8_t;
#line 60
using ::uint16_t;
#line 61
using ::uint32_t;
#line 62
using ::uint64_t;
#line 64
using ::int_least8_t;
#line 65
using ::int_least16_t;
#line 66
using ::int_least32_t;
#line 67
using ::int_least64_t;
#line 68
using ::uint_least8_t;
#line 69
using ::uint_least16_t;
#line 70
using ::uint_least32_t;
#line 71
using ::uint_least64_t;
#line 73
using ::int_fast8_t;
#line 74
using ::int_fast16_t;
#line 75
using ::int_fast32_t;
#line 76
using ::int_fast64_t;
#line 77
using ::uint_fast8_t;
#line 78
using ::uint_fast16_t;
#line 79
using ::uint_fast32_t;
#line 80
using ::uint_fast64_t;
#line 82
using ::intmax_t;
#line 83
using ::intptr_t;
#line 84
using ::uintmax_t;
#line 85
using ::uintptr_t;
#line 86
}
#line 88 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cstdint"
}
#line 92
#pragma warning(pop)
#pragma pack ( pop )
#line 13 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\cstring"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 20
namespace std { 
#pragma warning(push)
#pragma warning(disable : 4995)
#line 25
using ::memchr;
#line 26
using ::memcmp;
#line 27
using ::memcpy;
#line 28
using ::memmove;
#line 29
using ::memset;
#line 30
using ::strcat;
#line 31
using ::strchr;
#line 32
using ::strcmp;
#line 33
using ::strcoll;
#line 34
using ::strcpy;
#line 35
using ::strcspn;
#line 36
using ::strerror;
#line 37
using ::strlen;
#line 38
using ::strncat;
#line 39
using ::strncmp;
#line 40
using ::strncpy;
#line 41
using ::strpbrk;
#line 42
using ::strrchr;
#line 43
using ::strspn;
#line 44
using ::strstr;
#line 45
using ::strtok;
#line 46
using ::strxfrm;
#line 48
#pragma warning(pop)
}
#line 53
#pragma warning(pop)
#pragma pack ( pop )
#line 15 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 30
namespace std { 
#line 31
template< class > constexpr bool 
#line 32
_Always_false = false; 
#line 34
template< bool _First_value, class _First, class ..._Rest> 
#line 35
struct _Conjunction { 
#line 36
using type = _First; 
#line 37
}; 
#line 39
template< class _True, class _Next, class ..._Rest> 
#line 40
struct _Conjunction< true, _True, _Next, _Rest...>  { 
#line 41
using type = typename std::_Conjunction< static_cast< bool>(_Next::value), _Next, _Rest...> ::type; 
#line 42
}; 
#line 44
template< class ..._Traits> 
#line 45
struct conjunction : public true_type { }; 
#line 47
template< class _First, class ..._Rest> 
#line 48
struct conjunction< _First, _Rest...>  : public _Conjunction< static_cast< bool>(_First::value), _First, _Rest...> ::type { 
#line 50
}; 
#line 52
template< class ..._Traits> constexpr bool 
#line 53
conjunction_v = (conjunction< _Traits...> ::value); 
#line 55
template< class _Trait> 
#line 56
struct negation : public bool_constant< !(static_cast< bool>(_Trait::value))>  { }; 
#line 58
template< class _Trait> constexpr bool 
#line 59
negation_v = (negation< _Trait> ::value); 
#line 61
template< class _Ty> constexpr bool 
#line 62
is_void_v = is_same_v< remove_cv_t< _Ty> , void> ; 
#line 64
template< class _Ty> 
#line 65
struct is_void : public bool_constant< is_void_v< _Ty> >  { }; 
#line 67
template< class ..._Types> using void_t = void; 
#line 70
template< class _Ty> 
#line 71
struct _Identity { 
#line 72
using type = _Ty; 
#line 73
}; 
#line 74
template< class _Ty> using _Identity_t = typename _Identity< _Ty> ::type; 
#line 78
template< class _Ty> 
#line 79
struct add_const { 
#line 80
using type = const _Ty; 
#line 81
}; 
#line 83
template< class _Ty> using add_const_t = typename add_const< _Ty> ::type; 
#line 86
template< class _Ty> 
#line 87
struct add_volatile { 
#line 88
using type = volatile _Ty; 
#line 89
}; 
#line 91
template< class _Ty> using add_volatile_t = typename add_volatile< _Ty> ::type; 
#line 94
template< class _Ty> 
#line 95
struct add_cv { 
#line 96
using type = const volatile _Ty; 
#line 97
}; 
#line 99
template< class _Ty> using add_cv_t = typename add_cv< _Ty> ::type; 
#line 102
template< class _Ty, class  = void> 
#line 103
struct _Add_reference { 
#line 104
using _Lvalue = _Ty; 
#line 105
using _Rvalue = _Ty; 
#line 106
}; 
#line 108
template< class _Ty> 
#line 109
struct _Add_reference< _Ty, void_t< _Ty &> >  { 
#line 110
using _Lvalue = _Ty &; 
#line 111
using _Rvalue = _Ty &&; 
#line 112
}; 
#line 114
template< class _Ty> 
#line 115
struct add_lvalue_reference { 
#line 116
using type = typename _Add_reference< _Ty> ::_Lvalue; 
#line 117
}; 
#line 119
template< class _Ty> using add_lvalue_reference_t = typename _Add_reference< _Ty> ::_Lvalue; 
#line 122
template< class _Ty> 
#line 123
struct add_rvalue_reference { 
#line 124
using type = typename _Add_reference< _Ty> ::_Rvalue; 
#line 125
}; 
#line 127
template< class _Ty> using add_rvalue_reference_t = typename _Add_reference< _Ty> ::_Rvalue; 
#line 130
template < class _Ty >
add_rvalue_reference_t < _Ty > declval ( ) noexcept {
    static_assert ( false, "Calling declval is ill-formed, see N4950 [declval]/2." );
}
#line 135
template< class _Ty> 
#line 136
struct remove_extent { 
#line 137
using type = _Ty; 
#line 138
}; 
#line 140
template< class _Ty, size_t _Ix> 
#line 141
struct remove_extent< _Ty [_Ix]>  { 
#line 142
using type = _Ty; 
#line 143
}; 
#line 145
template< class _Ty> 
#line 146
struct remove_extent< _Ty []>  { 
#line 147
using type = _Ty; 
#line 148
}; 
#line 150
template< class _Ty> using remove_extent_t = typename remove_extent< _Ty> ::type; 
#line 153
template< class _Ty> 
#line 154
struct remove_all_extents { 
#line 155
using type = _Ty; 
#line 156
}; 
#line 158
template< class _Ty, size_t _Ix> 
#line 159
struct remove_all_extents< _Ty [_Ix]>  { 
#line 160
using type = typename std::remove_all_extents< _Ty> ::type; 
#line 161
}; 
#line 163
template< class _Ty> 
#line 164
struct remove_all_extents< _Ty []>  { 
#line 165
using type = typename std::remove_all_extents< _Ty> ::type; 
#line 166
}; 
#line 168
template< class _Ty> using remove_all_extents_t = typename remove_all_extents< _Ty> ::type; 
#line 171
template< class _Ty> 
#line 172
struct remove_pointer { 
#line 173
using type = _Ty; 
#line 174
}; 
#line 176
template< class _Ty> 
#line 177
struct remove_pointer< _Ty *>  { 
#line 178
using type = _Ty; 
#line 179
}; 
#line 181
template< class _Ty> 
#line 182
struct remove_pointer< _Ty *const>  { 
#line 183
using type = _Ty; 
#line 184
}; 
#line 186
template< class _Ty> 
#line 187
struct remove_pointer< _Ty *volatile>  { 
#line 188
using type = _Ty; 
#line 189
}; 
#line 191
template< class _Ty> 
#line 192
struct remove_pointer< _Ty *const volatile>  { 
#line 193
using type = _Ty; 
#line 194
}; 
#line 196
template< class _Ty> using remove_pointer_t = typename remove_pointer< _Ty> ::type; 
#line 199
template< class _Ty, class  = void> 
#line 200
struct _Add_pointer { 
#line 201
using type = _Ty; 
#line 202
}; 
#line 204
template< class _Ty> 
#line 205
struct _Add_pointer< _Ty, void_t< remove_reference_t< _Ty>  *> >  { 
#line 206
using type = remove_reference_t< _Ty>  *; 
#line 207
}; 
#line 209
template< class _Ty> 
#line 210
struct add_pointer { 
#line 211
using type = typename _Add_pointer< _Ty> ::type; 
#line 212
}; 
#line 214
template< class _Ty> using add_pointer_t = typename _Add_pointer< _Ty> ::type; 
#line 217
template< class > constexpr bool 
#line 218
is_array_v = false; 
#line 220
template< class _Ty, size_t _Nx> constexpr bool 
#line 221
is_array_v< _Ty [_Nx]>  = true; 
#line 223
template< class _Ty> constexpr bool 
#line 224
is_array_v< _Ty []>  = true; 
#line 226
template< class _Ty> 
#line 227
struct is_array : public bool_constant< is_array_v< _Ty> >  { }; 
#line 249 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class > constexpr bool 
#line 250
is_lvalue_reference_v = false; 
#line 252
template< class _Ty> constexpr bool 
#line 253
is_lvalue_reference_v< _Ty &>  = true; 
#line 255
template< class _Ty> 
#line 256
struct is_lvalue_reference : public bool_constant< is_lvalue_reference_v< _Ty> >  { }; 
#line 258
template< class > constexpr bool 
#line 259
is_rvalue_reference_v = false; 
#line 261
template< class _Ty> constexpr bool 
#line 262
is_rvalue_reference_v< _Ty &&>  = true; 
#line 264
template< class _Ty> 
#line 265
struct is_rvalue_reference : public bool_constant< is_rvalue_reference_v< _Ty> >  { }; 
#line 267
template< class > constexpr bool 
#line 268
is_reference_v = false; 
#line 270
template< class _Ty> constexpr bool 
#line 271
is_reference_v< _Ty &>  = true; 
#line 273
template< class _Ty> constexpr bool 
#line 274
is_reference_v< _Ty &&>  = true; 
#line 276
template< class _Ty> 
#line 277
struct is_reference : public bool_constant< is_reference_v< _Ty> >  { }; 
#line 279
template< class > constexpr bool 
#line 280
is_pointer_v = false; 
#line 282
template< class _Ty> constexpr bool 
#line 283
is_pointer_v< _Ty *>  = true; 
#line 285
template< class _Ty> constexpr bool 
#line 286
is_pointer_v< _Ty *const>  = true; 
#line 288
template< class _Ty> constexpr bool 
#line 289
is_pointer_v< _Ty *volatile>  = true; 
#line 291
template< class _Ty> constexpr bool 
#line 292
is_pointer_v< _Ty *const volatile>  = true; 
#line 294
template< class _Ty> 
#line 295
struct is_pointer : public bool_constant< is_pointer_v< _Ty> >  { }; 
#line 297
template< class _Ty> constexpr bool 
#line 298
is_null_pointer_v = is_same_v< remove_cv_t< _Ty> , std::nullptr_t> ; 
#line 301
template< class _Ty> 
#line 302
struct is_null_pointer : public bool_constant< is_null_pointer_v< _Ty> >  { }; 
#line 304
template< class _Ty> 
#line 305
struct is_union : public bool_constant< __is_union(_Ty)>  { }; 
#line 307
template< class _Ty> constexpr bool 
#line 308
is_union_v = __is_union(_Ty); 
#line 310
template< class _Ty> 
#line 311
struct is_class : public bool_constant< __is_class(_Ty)>  { }; 
#line 313
template< class _Ty> constexpr bool 
#line 314
is_class_v = __is_class(_Ty); 
#line 316
template< class _Ty> constexpr bool 
#line 317
is_fundamental_v = (is_arithmetic_v< _Ty>  || is_void_v< _Ty> ) || is_null_pointer_v< _Ty> ; 
#line 319
template< class _Ty> 
#line 320
struct is_fundamental : public bool_constant< is_fundamental_v< _Ty> >  { }; 
#line 322
template< class _From, class _To> 
#line 323
struct is_convertible : public bool_constant< __is_convertible_to(_From, _To)>  { 
#line 325
}; 
#line 327
template< class _From, class _To> constexpr bool 
#line 328
is_convertible_v = __is_convertible_to(_From, _To); 
#line 331
template< class _Ty> 
#line 332
struct is_convertible< _Ty &, volatile _Ty &>  : public true_type { }; 
#line 334
template< class _Ty> 
#line 335
struct is_convertible< volatile _Ty &, volatile _Ty &>  : public true_type { }; 
#line 337
template< class _Ty> 
#line 338
struct is_convertible< _Ty &, const volatile _Ty &>  : public true_type { }; 
#line 340
template< class _Ty> 
#line 341
struct is_convertible< volatile _Ty &, const volatile _Ty &>  : public true_type { }; 
#line 343
template< class _Ty> constexpr bool 
#line 344
is_convertible_v< _Ty &, volatile _Ty &>  = true; 
#line 346
template< class _Ty> constexpr bool 
#line 347
is_convertible_v< volatile _Ty &, volatile _Ty &>  = true; 
#line 349
template< class _Ty> constexpr bool 
#line 350
is_convertible_v< _Ty &, const volatile _Ty &>  = true; 
#line 352
template< class _Ty> constexpr bool 
#line 353
is_convertible_v< volatile _Ty &, const volatile _Ty &>  = true; 
#line 356 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 357
struct is_enum : public bool_constant< __is_enum(_Ty)>  { }; 
#line 359
template< class _Ty> constexpr bool 
#line 360
is_enum_v = __is_enum(_Ty); 
#line 370 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 371
struct is_compound : public bool_constant< !is_fundamental_v< _Ty> >  { }; 
#line 373
template< class _Ty> constexpr bool 
#line 374
is_compound_v = (!is_fundamental_v< _Ty> ); 
#line 499 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class ..._Types> 
#line 500
struct _Arg_types { }; 
#line 502
template< class _Ty1> 
#line 503
struct _Arg_types< _Ty1>  { 
#line 504
using argument_type = _Ty1; 
#line 505
}; 
#line 507
template< class _Ty1, class _Ty2> 
#line 508
struct _Arg_types< _Ty1, _Ty2>  { 
#line 509
using first_argument_type = _Ty1; 
#line 510
using second_argument_type = _Ty2; 
#line 511
}; 
#line 513
template< class _Ty> 
#line 514
struct _Is_memfunptr { 
#line 515
using _Bool_type = false_type; 
#line 516
}; 
#line 528
template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...)>  : public _Arg_types< _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...)>  : public _Arg_types< _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) const>  : public _Arg_types< const _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) const>  : public _Arg_types< const _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) volatile>  : public _Arg_types< volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) volatile>  : public _Arg_types< volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) const volatile>  : public _Arg_types< const volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) const volatile>  : public _Arg_types< const volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) &>  : public _Arg_types< _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) &>  : public _Arg_types< _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) const &>  : public _Arg_types< const _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) const &>  : public _Arg_types< const _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) volatile &>  : public _Arg_types< volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) volatile &>  : public _Arg_types< volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) const volatile &>  : public _Arg_types< const volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) const volatile &>  : public _Arg_types< const volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< true, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) &&>  : public _Arg_types< _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) &&>  : public _Arg_types< _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) const &&>  : public _Arg_types< const _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) const &&>  : public _Arg_types< const _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) volatile &&>  : public _Arg_types< volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) volatile &&>  : public _Arg_types< volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__cdecl _Arg0::*)(_Types ...) const volatile &&>  : public _Arg_types< const volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false, _Ret (_Types ...)> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (__vectorcall _Arg0::*)(_Types ...) const volatile &&>  : public _Arg_types< const volatile _Arg0 *, _Types...>  { using _Bool_type = std::true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false, _Ret (_Types ...)> ; }; 
#line 541
template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...)>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) const>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) volatile>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) const volatile>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) &>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) const &>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) volatile &>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) const volatile &>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) &&>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) const &&>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) volatile &&>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; template< class _Ret, class _Arg0, class ..._Types> struct _Is_memfunptr< _Ret (_Arg0::*)(_Types ..., ...) const volatile &&>  { using _Bool_type = true_type; using result_type = _Ret; using _Class_type = _Arg0; using _Guide_type = enable_if< false> ; }; 
#line 564 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> constexpr bool 
#line 565
is_member_function_pointer_v = (_Is_memfunptr< remove_cv_t< _Ty> > ::_Bool_type::value); 
#line 568 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 569
struct is_member_function_pointer : public bool_constant< is_member_function_pointer_v< _Ty> >  { }; 
#line 571
template< class > constexpr bool 
#line 572
is_const_v = false; 
#line 574
template< class _Ty> constexpr bool 
#line 575
is_const_v< const _Ty>  = true; 
#line 577
template< class _Ty> 
#line 578
struct is_const : public bool_constant< is_const_v< _Ty> >  { }; 
#line 580
template< class > constexpr bool 
#line 581
is_volatile_v = false; 
#line 583
template< class _Ty> constexpr bool 
#line 584
is_volatile_v< volatile _Ty>  = true; 
#line 586
template< class _Ty> 
#line 587
struct is_volatile : public bool_constant< is_volatile_v< _Ty> >  { }; 
#line 589
template< class _Ty> constexpr bool 
#line 590
is_function_v = ((!is_const_v< const _Ty> ) && (!is_reference_v< _Ty> )); 
#line 593
template< class _Ty> 
#line 594
struct is_function : public bool_constant< is_function_v< _Ty> >  { }; 
#line 596
template< class _Ty> constexpr bool 
#line 597
is_object_v = is_const_v< const _Ty>  && (!is_void_v< _Ty> ); 
#line 600
template< class _Ty> 
#line 601
struct is_object : public bool_constant< is_object_v< _Ty> >  { }; 
#line 603
template< class > 
#line 604
struct _Is_member_object_pointer { 
#line 605
static constexpr bool value = false; 
#line 606
}; 
#line 608
template< class _Ty1, class _Ty2> 
#line 609
struct _Is_member_object_pointer< _Ty1 _Ty2::*>  { 
#line 610
static constexpr bool value = (!is_function_v< _Ty1> ); 
#line 611
using _Class_type = _Ty2; 
#line 612
}; 
#line 618
template< class _Ty> constexpr bool 
#line 619
is_member_object_pointer_v = (_Is_member_object_pointer< remove_cv_t< _Ty> > ::value); 
#line 622 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 623
struct is_member_object_pointer : public bool_constant< is_member_object_pointer_v< _Ty> >  { }; 
#line 629
template< class _Ty> constexpr bool 
#line 630
is_member_pointer_v = is_member_object_pointer_v< _Ty>  || is_member_function_pointer_v< _Ty> ; 
#line 633 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 634
struct is_member_pointer : public bool_constant< is_member_pointer_v< _Ty> >  { }; 
#line 636
template< class _Ty> constexpr bool 
#line 637
is_scalar_v = (((is_arithmetic_v< _Ty>  || is_enum_v< _Ty> ) || is_pointer_v< _Ty> ) || is_member_pointer_v< _Ty> ) || is_null_pointer_v< _Ty> ; 
#line 640
template< class _Ty> 
#line 641
struct is_scalar : public bool_constant< is_scalar_v< _Ty> >  { }; 
#line 643
template< class _Ty> 
#line 644
struct is_pod : public bool_constant< __is_pod(_Ty)>  { }; 
#line 646
template< class _Ty> constexpr bool 
#line 647
is_pod_v = __is_pod(_Ty); 
#line 649
template< class _Ty> 
#line 650
struct is_empty : public bool_constant< __is_empty(_Ty)>  { }; 
#line 652
template< class _Ty> constexpr bool 
#line 653
is_empty_v = __is_empty(_Ty); 
#line 655
template< class _Ty> 
#line 656
struct is_polymorphic : public bool_constant< __is_polymorphic(_Ty)>  { }; 
#line 658
template< class _Ty> constexpr bool 
#line 659
is_polymorphic_v = __is_polymorphic(_Ty); 
#line 661
template< class _Ty> 
#line 662
struct is_abstract : public bool_constant< __is_abstract(_Ty)>  { }; 
#line 664
template< class _Ty> constexpr bool 
#line 665
is_abstract_v = __is_abstract(_Ty); 
#line 667
template< class _Ty> 
#line 668
struct is_final : public bool_constant< __is_final(_Ty)>  { }; 
#line 670
template< class _Ty> constexpr bool 
#line 671
is_final_v = __is_final(_Ty); 
#line 673
template< class _Ty> 
#line 674
struct is_standard_layout : public bool_constant< __is_standard_layout(_Ty)>  { }; 
#line 676
template< class _Ty> constexpr bool 
#line 677
is_standard_layout_v = __is_standard_layout(_Ty); 
#line 680
template< class _Ty> 
#line 681
struct is_literal_type : public bool_constant< __is_literal_type(_Ty)>  { 
#line 683
}; 
#line 685
template< class _Ty> constexpr bool 
#line 686
is_literal_type_v = __is_literal_type(_Ty); 
#line 689 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 690
struct is_trivial : public bool_constant< __is_trivial(_Ty)>  { }; 
#line 692
template< class _Ty> constexpr bool 
#line 693
is_trivial_v = __is_trivial(_Ty); 
#line 695
template< class _Ty> 
#line 696
struct is_trivially_copyable : public bool_constant< __is_trivially_copyable(_Ty)>  { 
#line 698
}; 
#line 700
template< class _Ty> constexpr bool 
#line 701
is_trivially_copyable_v = __is_trivially_copyable(_Ty); 
#line 703
template< class _Ty> 
#line 704
struct has_virtual_destructor : public bool_constant< __has_virtual_destructor(_Ty)>  { 
#line 706
}; 
#line 708
template< class _Ty> constexpr bool 
#line 709
has_virtual_destructor_v = __has_virtual_destructor(_Ty); 
#line 738 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty, class ..._Args> 
#line 739
struct is_constructible : public bool_constant< __is_constructible(_Ty, _Args...)>  { 
#line 741
}; 
#line 743
template< class _Ty, class ..._Args> constexpr bool 
#line 744
is_constructible_v = __is_constructible(_Ty, _Args...); 
#line 746
template< class _Ty> 
#line 747
struct is_copy_constructible : public bool_constant< __is_constructible(_Ty, add_lvalue_reference_t< const _Ty> )>  { 
#line 749
}; 
#line 751
template< class _Ty> constexpr bool 
#line 752
is_copy_constructible_v = __is_constructible(_Ty, add_lvalue_reference_t< const _Ty> ); 
#line 754
template< class _Ty> 
#line 755
struct is_default_constructible : public bool_constant< __is_constructible(_Ty)>  { 
#line 757
}; 
#line 759
template< class _Ty> constexpr bool 
#line 760
is_default_constructible_v = __is_constructible(_Ty); 
#line 762
template< class _Ty, class  = void> 
#line 763
struct _Is_implicitly_default_constructible : public false_type { 
#line 765
}; 
#line 767
template< class _Ty> void _Implicitly_default_construct(const _Ty &); 
#line 770
template< class _Ty> 
#line 771
struct _Is_implicitly_default_constructible< _Ty, void_t< decltype(_Implicitly_default_construct< _Ty> ({}))> >  : public true_type { 
#line 772
}; 
#line 774
template< class _Ty> 
#line 775
struct is_move_constructible : public bool_constant< __is_constructible(_Ty, _Ty)>  { 
#line 777
}; 
#line 779
template< class _Ty> constexpr bool 
#line 780
is_move_constructible_v = __is_constructible(_Ty, _Ty); 
#line 782
template< class _To, class _From> 
#line 783
struct is_assignable : public bool_constant< __is_assignable(_To, _From)>  { }; 
#line 785
template< class _To, class _From> constexpr bool 
#line 786
is_assignable_v = __is_assignable(_To, _From); 
#line 792 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _To, class _From> using _Is_assignable_no_precondition_check = is_assignable< _To, _From> ; 
#line 796 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 797
struct is_copy_assignable : public bool_constant< __is_assignable(add_lvalue_reference_t< _Ty> , add_lvalue_reference_t< const _Ty> )>  { 
#line 800
}; 
#line 802
template< class _Ty> constexpr bool 
#line 803
is_copy_assignable_v = __is_assignable(add_lvalue_reference_t< _Ty> , add_lvalue_reference_t< const _Ty> ); 
#line 815 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> using _Is_copy_assignable_no_precondition_check = is_copy_assignable< _Ty> ; 
#line 818
template< class _Ty> constexpr bool 
#line 819
_Is_copy_assignable_unchecked_v = is_copy_assignable_v< _Ty> ; 
#line 822 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 823
struct is_move_assignable : public bool_constant< __is_assignable(add_lvalue_reference_t< _Ty> , _Ty)>  { 
#line 825
}; 
#line 827
template< class _Ty> constexpr bool 
#line 828
is_move_assignable_v = __is_assignable(add_lvalue_reference_t< _Ty> , _Ty); 
#line 839 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> using _Is_move_assignable_no_precondition_check = is_move_assignable< _Ty> ; 
#line 842
template< class _Ty> constexpr bool 
#line 843
_Is_move_assignable_unchecked_v = is_move_assignable_v< _Ty> ; 
#line 846 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 847
struct is_destructible : public bool_constant< __is_destructible(_Ty)>  { 
#line 849
}; 
#line 851
template< class _Ty> constexpr bool 
#line 852
is_destructible_v = __is_destructible(_Ty); 
#line 854
template< class _Ty, class ..._Args> 
#line 855
struct is_trivially_constructible : public bool_constant< __is_trivially_constructible(_Ty, _Args...)>  { 
#line 857
}; 
#line 859
template< class _Ty, class ..._Args> constexpr bool 
#line 860
is_trivially_constructible_v = __is_trivially_constructible(_Ty, _Args...); 
#line 862
template< class _Ty> 
#line 863
struct is_trivially_copy_constructible : public bool_constant< __is_trivially_constructible(_Ty, add_lvalue_reference_t< const _Ty> )>  { 
#line 866
}; 
#line 868
template< class _Ty> constexpr bool 
#line 869
is_trivially_copy_constructible_v = __is_trivially_constructible(_Ty, add_lvalue_reference_t< const _Ty> ); 
#line 871
template< class _Ty> 
#line 872
struct is_trivially_default_constructible : public bool_constant< __is_trivially_constructible(_Ty)>  { 
#line 874
}; 
#line 876
template< class _Ty> constexpr bool 
#line 877
is_trivially_default_constructible_v = __is_trivially_constructible(_Ty); 
#line 879
template< class _Ty> 
#line 880
struct is_trivially_move_constructible : public bool_constant< __is_trivially_constructible(_Ty, _Ty)>  { 
#line 882
}; 
#line 884
template< class _Ty> constexpr bool 
#line 885
is_trivially_move_constructible_v = __is_trivially_constructible(_Ty, _Ty); 
#line 887
template< class _To, class _From> 
#line 888
struct is_trivially_assignable : public bool_constant< __is_trivially_assignable(_To, _From)>  { 
#line 890
}; 
#line 892
template< class _To, class _From> constexpr bool 
#line 893
is_trivially_assignable_v = __is_trivially_assignable(_To, _From); 
#line 895
template< class _Ty> 
#line 896
struct is_trivially_copy_assignable : public bool_constant< __is_trivially_assignable(add_lvalue_reference_t< _Ty> , add_lvalue_reference_t< const _Ty> )>  { 
#line 899
}; 
#line 901
template< class _Ty> constexpr bool 
#line 902
is_trivially_copy_assignable_v = __is_trivially_assignable(add_lvalue_reference_t< _Ty> , add_lvalue_reference_t< const _Ty> ); 
#line 905
template< class _Ty> 
#line 906
struct is_trivially_move_assignable : public bool_constant< __is_trivially_assignable(add_lvalue_reference_t< _Ty> , _Ty)>  { 
#line 908
}; 
#line 910
template< class _Ty> constexpr bool 
#line 911
is_trivially_move_assignable_v = __is_trivially_assignable(add_lvalue_reference_t< _Ty> , _Ty); 
#line 913
template< class _Ty> 
#line 914
struct is_trivially_destructible : public bool_constant< __is_trivially_destructible(_Ty)>  { 
#line 916
}; 
#line 918
template< class _Ty> constexpr bool 
#line 919
is_trivially_destructible_v = __is_trivially_destructible(_Ty); 
#line 921
template< class _Ty, class ..._Args> 
#line 922
struct is_nothrow_constructible : public bool_constant< __is_nothrow_constructible(_Ty, _Args...)>  { 
#line 924
}; 
#line 926
template< class _Ty, class ..._Args> constexpr bool 
#line 927
is_nothrow_constructible_v = __is_nothrow_constructible(_Ty, _Args...); 
#line 929
template< class _Ty> 
#line 930
struct is_nothrow_copy_constructible : public bool_constant< __is_nothrow_constructible(_Ty, add_lvalue_reference_t< const _Ty> )>  { 
#line 934
}; 
#line 936
template< class _Ty> constexpr bool 
#line 937
is_nothrow_copy_constructible_v = __is_nothrow_constructible(_Ty, add_lvalue_reference_t< const _Ty> ); 
#line 939
template< class _Ty> 
#line 940
struct is_nothrow_default_constructible : public bool_constant< __is_nothrow_constructible(_Ty)>  { 
#line 942
}; 
#line 944
template< class _Ty> constexpr bool 
#line 945
is_nothrow_default_constructible_v = __is_nothrow_constructible(_Ty); 
#line 947
template< class _Ty> 
#line 948
struct is_nothrow_move_constructible : public bool_constant< __is_nothrow_constructible(_Ty, _Ty)>  { 
#line 950
}; 
#line 952
template< class _Ty> constexpr bool 
#line 953
is_nothrow_move_constructible_v = __is_nothrow_constructible(_Ty, _Ty); 
#line 955
template< class _To, class _From> 
#line 956
struct is_nothrow_assignable : public bool_constant< __is_nothrow_assignable(_To, _From)>  { 
#line 958
}; 
#line 960
template< class _To, class _From> constexpr bool 
#line 961
is_nothrow_assignable_v = __is_nothrow_assignable(_To, _From); 
#line 963
template< class _Ty> 
#line 964
struct is_nothrow_copy_assignable : public bool_constant< __is_nothrow_assignable(add_lvalue_reference_t< _Ty> , add_lvalue_reference_t< const _Ty> )>  { 
#line 967
}; 
#line 969
template< class _Ty> constexpr bool 
#line 970
is_nothrow_copy_assignable_v = __is_nothrow_assignable(add_lvalue_reference_t< _Ty> , add_lvalue_reference_t< const _Ty> ); 
#line 973
template< class _Ty> 
#line 974
struct is_nothrow_move_assignable : public bool_constant< __is_nothrow_assignable(add_lvalue_reference_t< _Ty> , _Ty)>  { 
#line 976
}; 
#line 978
template< class _Ty> constexpr bool 
#line 979
is_nothrow_move_assignable_v = __is_nothrow_assignable(add_lvalue_reference_t< _Ty> , _Ty); 
#line 981
template< class _Ty> 
#line 982
struct is_nothrow_destructible : public bool_constant< __is_nothrow_destructible(_Ty)>  { 
#line 985
}; 
#line 987
template< class _Ty> constexpr bool 
#line 988
is_nothrow_destructible_v = __is_nothrow_destructible(_Ty); 
#line 990
template< class _Ty, bool  = is_integral_v< _Ty> > 
#line 991
struct _Sign_base { 
#line 992
using _Uty = remove_cv_t< _Ty> ; 
#line 994
static constexpr bool _Signed = ((static_cast< _Uty>(-1)) < (static_cast< _Uty>(0))); 
#line 995
static constexpr bool _Unsigned = (!_Signed); 
#line 996
}; 
#line 998
template< class _Ty> 
#line 999
struct _Sign_base< _Ty, false>  { 
#line 1001
static constexpr bool _Signed = is_floating_point_v< _Ty> ; 
#line 1002
static constexpr bool _Unsigned = false; 
#line 1003
}; 
#line 1005
template< class _Ty> 
#line 1006
struct is_signed : public bool_constant< _Sign_base< _Ty> ::_Signed>  { }; 
#line 1008
template< class _Ty> constexpr bool 
#line 1009
is_signed_v = (_Sign_base< _Ty> ::_Signed); 
#line 1011
template< class _Ty> 
#line 1012
struct is_unsigned : public bool_constant< _Sign_base< _Ty> ::_Unsigned>  { }; 
#line 1014
template< class _Ty> constexpr bool 
#line 1015
is_unsigned_v = (_Sign_base< _Ty> ::_Unsigned); 
#line 1017
template< class _Ty> constexpr bool 
#line 1018
_Is_nonbool_integral = is_integral_v< _Ty>  && (!is_same_v< remove_cv_t< _Ty> , bool> ); 
#line 1020
template< bool > 
#line 1021
struct _Select { 
#line 1022
template< class _Ty1, class > using _Apply = _Ty1; 
#line 1024
}; 
#line 1027
template<> struct _Select< false>  { 
#line 1028
template< class , class _Ty2> using _Apply = _Ty2; 
#line 1030
}; 
#line 1032
template< size_t > struct _Make_signed2; 
#line 1036
template<> struct _Make_signed2< 1Ui64>  { 
#line 1037
template< class > using _Apply = signed char; 
#line 1039
}; 
#line 1042
template<> struct _Make_signed2< 2Ui64>  { 
#line 1043
template< class > using _Apply = short; 
#line 1045
}; 
#line 1048
template<> struct _Make_signed2< 4Ui64>  { 
#line 1049
template< class _Ty> using _Apply = typename _Select< is_same_v< _Ty, long>  || is_same_v< _Ty, unsigned long> > ::template _Apply< long, int> ; 
#line 1052
}; 
#line 1055
template<> struct _Make_signed2< 8Ui64>  { 
#line 1056
template< class > using _Apply = __int64; 
#line 1058
}; 
#line 1060
template< class _Ty> using _Make_signed1 = typename _Make_signed2< sizeof(_Ty)> ::template _Apply< _Ty> ; 
#line 1064
template< class _Ty> 
#line 1065
struct make_signed { 
#line 1066
static_assert((_Is_nonbool_integral< _Ty>  || is_enum_v< _Ty> ), "make_signed<T> requires that T shall be a (possibly cv-qualified) integral type or enumeration but not a bool type.");
#line 1070
using type = typename remove_cv< _Ty> ::template _Apply< _Make_signed1> ; 
#line 1071
}; 
#line 1073
template< class _Ty> using make_signed_t = typename make_signed< _Ty> ::type; 
#line 1076
template< size_t > struct _Make_unsigned2; 
#line 1080
template<> struct _Make_unsigned2< 1Ui64>  { 
#line 1081
template< class > using _Apply = unsigned char; 
#line 1083
}; 
#line 1086
template<> struct _Make_unsigned2< 2Ui64>  { 
#line 1087
template< class > using _Apply = unsigned short; 
#line 1089
}; 
#line 1092
template<> struct _Make_unsigned2< 4Ui64>  { 
#line 1093
template< class _Ty> using _Apply = typename _Select< is_same_v< _Ty, long>  || is_same_v< _Ty, unsigned long> > ::template _Apply< unsigned long, unsigned> ; 
#line 1097
}; 
#line 1100
template<> struct _Make_unsigned2< 8Ui64>  { 
#line 1101
template< class > using _Apply = unsigned __int64; 
#line 1103
}; 
#line 1105
template< class _Ty> using _Make_unsigned1 = typename _Make_unsigned2< sizeof(_Ty)> ::template _Apply< _Ty> ; 
#line 1109
template< class _Ty> 
#line 1110
struct make_unsigned { 
#line 1111
static_assert((_Is_nonbool_integral< _Ty>  || is_enum_v< _Ty> ), "make_unsigned<T> requires that T shall be a (possibly cv-qualified) integral type or enumeration but not a bool type.");
#line 1115
using type = typename remove_cv< _Ty> ::template _Apply< _Make_unsigned1> ; 
#line 1116
}; 
#line 1118
template< class _Ty> using make_unsigned_t = typename make_unsigned< _Ty> ::type; 
#line 1121
template < class _Rep >
constexpr make_unsigned_t < _Rep > _Unsigned_value ( _Rep _Val ) {
    return static_cast < make_unsigned_t < _Rep >> ( _Val );
}
#line 1126
template< class _Ty> 
#line 1127
struct alignment_of : public integral_constant< unsigned __int64, __alignof(_Ty)>  { }; 
#line 1129
template< class _Ty> constexpr size_t 
#line 1130
alignment_of_v = __alignof(_Ty); 
#line 1132
template< class _Ty, size_t _Len> 
#line 1133
union _Align_type { 
#line 1134
_Ty _Val; 
#line 1135
char _Pad[_Len]; 
#line 1136
}; 
#line 1138
template< size_t _Len, size_t _Align, class _Ty, bool _Ok> struct _Aligned; 
#line 1141
template< size_t _Len, size_t _Align, class _Ty> 
#line 1142
struct _Aligned< _Len, _Align, _Ty, true>  { 
#line 1143
using type = _Align_type< _Ty, _Len> ; 
#line 1144
}; 
#line 1146
template< size_t _Len, size_t _Align> 
#line 1147
struct _Aligned< _Len, _Align, double, false>  { 
#line 1154
static_assert((_Always_false< _Aligned> ), "You\'ve instantiated std::aligned_storage<Len, Align> with an extended alignment (in other words, Align > alignof(max_align_t))." " Before VS 2017 15.8, the member \"type\" would non-conformingly have an alignment of only alignof(max_align_t). VS 2017 15.8 wa" "s fixed to handle this correctly, but the fix inherently changes layout and breaks binary compatibility (*only* for uses of alig" "ned_storage with extended alignments). To suppress this error, please define either (1) _ENABLE_EXTENDED_ALIGNED_STORAGE to conf" "irm that you want a type with an extended alignment, or (2) _DISABLE_EXTENDED_ALIGNED_STORAGE to get the old non-conforming beha" "vior.");
#line 1164 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
using type = _Align_type< max_align_t, _Len> ; 
#line 1166 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
}; 
#line 1168
template< size_t _Len, size_t _Align> 
#line 1169
struct _Aligned< _Len, _Align, int, false>  { 
#line 1170
using _Next = double; 
#line 1171
static constexpr bool _Fits = (_Align <= __alignof(_Next)); 
#line 1172
using type = typename std::_Aligned< _Len, _Align, double, _Fits> ::type; 
#line 1173
}; 
#line 1175
template< size_t _Len, size_t _Align> 
#line 1176
struct _Aligned< _Len, _Align, short, false>  { 
#line 1177
using _Next = int; 
#line 1178
static constexpr bool _Fits = (_Align <= __alignof(_Next)); 
#line 1179
using type = typename std::_Aligned< _Len, _Align, int, _Fits> ::type; 
#line 1180
}; 
#line 1182
template< size_t _Len, size_t _Align> 
#line 1183
struct _Aligned< _Len, _Align, char, false>  { 
#line 1184
using _Next = short; 
#line 1185
static constexpr bool _Fits = (_Align <= __alignof(_Next)); 
#line 1186
using type = typename std::_Aligned< _Len, _Align, short, _Fits> ::type; 
#line 1187
}; 
#line 1190
template< size_t _Len, size_t _Align = 8Ui64> 
#line 1191
struct _Aligned_storage { 
#line 1192
using _Next = char; 
#line 1193
static constexpr bool _Fits = (_Align <= __alignof(_Next)); 
#line 1194
using type = typename _Aligned< _Len, _Align, char, _Fits> ::type; 
#line 1195
}; 
#line 1197
template< size_t _Len, size_t _Align = 8Ui64> using _Aligned_storage_t = typename _Aligned_storage< _Len, _Align> ::type; 
#line 1200
template< size_t _Len, size_t _Align = 8Ui64> 
#line 1201
struct aligned_storage { 
#line 1202
using type = _Aligned_storage_t< _Len, _Align> ; 
#line 1203
}; 
#line 1205
template< size_t _Len, size_t _Align = 8Ui64> using aligned_storage_t = _Aligned_storage_t< _Len, _Align> ; 
#line 1208
template< size_t ..._Vals> struct _Maximum; 
#line 1212
template<> struct _Maximum< >  : public integral_constant< unsigned __int64, 0Ui64>  { }; 
#line 1214
template< size_t _Val> 
#line 1215
struct _Maximum< _Val>  : public integral_constant< unsigned __int64, _Val>  { }; 
#line 1217
template< size_t _First, size_t _Second, size_t ..._Rest> 
#line 1218
struct _Maximum< _First, _Second, _Rest...>  : public std::_Maximum< (((_First) < _Second) ? _Second : _First), _Rest...> ::type { 
#line 1220
}; 
#line 1222
template< size_t _Len, class ..._Types> 
#line 1223
struct aligned_union { 
#line 1225
static constexpr size_t _Max_len = (_Maximum< _Len, sizeof(_Types)...> ::value); 
#line 1226
static constexpr size_t alignment_value = (_Maximum< __alignof(_Types)...> ::value); 
#line 1228
using type = _Aligned_storage_t< _Max_len, alignment_value> ; 
#line 1229
}; 
#line 1231
__pragma( warning(push)) __pragma( warning(disable : 4996)) 
#line 1232
template< size_t _Len, class ..._Types> using aligned_union_t = typename aligned_union< _Len, _Types...> ::type; 
#line 1234
__pragma( warning(pop)) 
#line 1236
template< class _Ty, bool  = is_enum_v< _Ty> > 
#line 1237
struct _Underlying_type { 
#line 1238
using type = __underlying_type(_Ty); 
#line 1239
}; 
#line 1241
template< class _Ty> 
#line 1242
struct _Underlying_type< _Ty, false>  { }; 
#line 1244
template< class _Ty> 
#line 1245
struct underlying_type : public _Underlying_type< _Ty>  { }; 
#line 1247
template< class _Ty> using underlying_type_t = typename _Underlying_type< _Ty> ::type; 
#line 1250
template< class _Ty> constexpr size_t 
#line 1251
rank_v = (0); 
#line 1253
template< class _Ty, size_t _Nx> constexpr size_t 
#line 1254
rank_v< _Ty [_Nx]>  = rank_v< _Ty>  + 1; 
#line 1256
template< class _Ty> constexpr size_t 
#line 1257
rank_v< _Ty []>  = rank_v< _Ty>  + 1; 
#line 1259
template< class _Ty> 
#line 1260
struct rank : public integral_constant< unsigned __int64, rank_v< _Ty> >  { }; 
#line 1262
template< class _Ty, unsigned _Ix = 0U> constexpr size_t 
#line 1263
extent_v = (0); 
#line 1265
template< class _Ty, size_t _Nx> constexpr size_t 
#line 1266
extent_v< _Ty [_Nx], 0>  = _Nx; 
#line 1268
template< class _Ty, unsigned _Ix, size_t _Nx> constexpr size_t 
#line 1269
extent_v< _Ty [_Nx], _Ix>  = extent_v< _Ty, _Ix - (1)> ; 
#line 1271
template< class _Ty, unsigned _Ix> constexpr size_t 
#line 1272
extent_v< _Ty [], _Ix>  = extent_v< _Ty, _Ix - (1)> ; 
#line 1274
template< class _Ty, unsigned _Ix = 0U> 
#line 1275
struct extent : public integral_constant< unsigned __int64, extent_v< _Ty, _Ix> >  { }; 
#line 1277
template< class _Base, class _Derived> 
#line 1278
struct is_base_of : public bool_constant< __is_base_of(_Base, _Derived)>  { 
#line 1280
}; 
#line 1282
template< class _Base, class _Derived> constexpr bool 
#line 1283
is_base_of_v = __is_base_of(_Base, _Derived); 
#line 1285
template< class _Ty> 
#line 1286
struct decay { 
#line 1287
using _Ty1 = remove_reference_t< _Ty> ; 
#line 1288
using _Ty2 = typename _Select< is_function_v< _Ty1> > ::template _Apply< add_pointer< _Ty1> , remove_cv< _Ty1> > ; 
#line 1289
using type = typename _Select< is_array_v< _Ty1> > ::template _Apply< add_pointer< remove_extent_t< _Ty1> > , _Ty2> ::type; 
#line 1290
}; 
#line 1292
template< class _Ty> using decay_t = typename decay< _Ty> ::type; 
#line 1295
template< class _Ty1, class _Ty2> using _Conditional_type = decltype(false ? std::declval< _Ty1> () : std::declval< _Ty2> ()); 
#line 1313 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty1, class _Ty2, class  = void> 
#line 1314
struct _Decayed_cond_oper { }; 
#line 1317 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty1, class _Ty2> 
#line 1318
struct _Decayed_cond_oper< _Ty1, _Ty2, void_t< _Conditional_type< _Ty1, _Ty2> > >  { 
#line 1319
using type = decay_t< _Conditional_type< _Ty1, _Ty2> > ; 
#line 1320
}; 
#line 1322
template< class ..._Ty> struct common_type; 
#line 1325
template< class ..._Ty> using common_type_t = typename common_type< _Ty...> ::type; 
#line 1329
template<> struct common_type< >  { }; 
#line 1331
template< class _Ty1> 
#line 1332
struct common_type< _Ty1>  : public std::common_type< _Ty1, _Ty1>  { }; 
#line 1334
template< class _Ty1, class _Ty2, class _Decayed1 = decay_t< _Ty1> , class _Decayed2 = decay_t< _Ty2> > 
#line 1335
struct _Common_type2 : public common_type< _Decayed1, _Decayed2>  { }; 
#line 1337
template< class _Ty1, class _Ty2> 
#line 1338
struct _Common_type2< _Ty1, _Ty2, _Ty1, _Ty2>  : public _Decayed_cond_oper< _Ty1, _Ty2>  { }; 
#line 1340
template< class _Ty1, class _Ty2> 
#line 1341
struct common_type< _Ty1, _Ty2>  : public _Common_type2< _Ty1, _Ty2>  { }; 
#line 1343
template< class _Void, class _Ty1, class _Ty2, class ..._Rest> 
#line 1344
struct _Common_type3 { }; 
#line 1346
template< class _Ty1, class _Ty2, class ..._Rest> 
#line 1347
struct _Common_type3< void_t< common_type_t< _Ty1, _Ty2> > , _Ty1, _Ty2, _Rest...>  : public common_type< common_type_t< _Ty1, _Ty2> , _Rest...>  { 
#line 1348
}; 
#line 1350
template< class _Ty1, class _Ty2, class ..._Rest> 
#line 1351
struct common_type< _Ty1, _Ty2, _Rest...>  : public _Common_type3< void, _Ty1, _Ty2, _Rest...>  { }; 
#line 1353
template< class _Ty> _Ty _Returns_exactly() noexcept; 
#line 1518 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Type, template< class ...>  class _Template> constexpr bool 
#line 1519
_Is_specialization_v = false; 
#line 1520
template< template< class ...>  class _Template, class ..._Types> constexpr bool 
#line 1521
_Is_specialization_v< _Template< _Types...> , _Template>  = true; 
#line 1523
template< class _Type, template< class ...>  class _Template> 
#line 1524
struct _Is_specialization : public bool_constant< _Is_specialization_v< _Type, _Template> >  { }; 
#line 1526
template< class _Ty> 
#line 1527
[[nodiscard]] constexpr _Ty &&forward(remove_reference_t< _Ty>  &_Arg) noexcept { 
#line 1528
return static_cast< _Ty &&>(_Arg); 
#line 1529
} 
#line 1531
template< class _Ty> 
#line 1532
[[nodiscard]] constexpr _Ty &&forward(remove_reference_t< _Ty>  &&_Arg) noexcept { 
#line 1533
static_assert((!is_lvalue_reference_v< _Ty> ), "bad forward call");
#line 1534
return static_cast< _Ty &&>(_Arg); 
#line 1535
} 
#line 1537
template< class _Ty> 
#line 1538
[[nodiscard]] constexpr remove_reference_t< _Ty>  &&move(_Ty &&_Arg) noexcept { 
#line 1539
return static_cast< remove_reference_t< _Ty>  &&>(_Arg); 
#line 1540
} 
#line 1542
template < class _Ty >
[ [ nodiscard ] ] constexpr
    conditional_t < ! is_nothrow_move_constructible_v < _Ty > && is_copy_constructible_v < _Ty >, const _Ty &, _Ty && >
    move_if_noexcept ( _Ty & _Arg ) noexcept {
    return :: std :: move ( _Arg );
}
#line 1549
template < class _Ty >
[ [ nodiscard ] ] constexpr _Ty * addressof ( _Ty & _Val ) noexcept {
    return __builtin_addressof ( _Val );
}
#line 1554
template < class _Ty >
const _Ty * addressof ( const _Ty && ) = delete;
#line 1557
#pragma warning(push)
#pragma warning(disable : 5215)
#pragma warning(disable : 5216)
#line 1566
template< class _Ty> 
#line 1567
[[nodiscard]] _Ty 
#line 1566
_Fake_copy_init(_Ty) noexcept; 
#line 1577
#pragma warning(pop)
#line 1579
template < class _Ty >
class reference_wrapper;
#line 1586
enum class _Invoker_strategy { 
#line 1587
_Functor, 
#line 1588
_Pmf_object, 
#line 1589
_Pmf_refwrap, 
#line 1590
_Pmf_pointer, 
#line 1591
_Pmd_object, 
#line 1592
_Pmd_refwrap, 
#line 1593
_Pmd_pointer
#line 1594
}; 
#line 1596
struct _Invoker_functor { 
#line 1597
static constexpr _Invoker_strategy _Strategy = _Invoker_strategy::_Functor; 
#line 1599
template < class _Callable, class ... _Types >
    static constexpr auto _Call ( _Callable && _Obj, _Types && ... _Args )
        noexcept ( noexcept ( static_cast < _Callable && > ( _Obj ) ( static_cast < _Types && > ( _Args ) ... ) ) )
        -> decltype ( static_cast < _Callable && > ( _Obj ) ( static_cast < _Types && > ( _Args ) ... ) ) {
        return static_cast < _Callable && > ( _Obj ) ( static_cast < _Types && > ( _Args ) ... );
    }
#line 1605
}; 
#line 1607
struct _Invoker_pmf_object { 
#line 1608
static constexpr _Invoker_strategy _Strategy = _Invoker_strategy::_Pmf_object; 
#line 1610
template < class _Decayed, class _Ty1, class ... _Types2 >
    static constexpr auto _Call ( _Decayed _Pmf, _Ty1 && _Arg1, _Types2 && ... _Args2 )
        noexcept ( noexcept ( ( static_cast < _Ty1 && > ( _Arg1 ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... ) ) )
        -> decltype ( ( static_cast < _Ty1 && > ( _Arg1 ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... ) ) {
        return ( static_cast < _Ty1 && > ( _Arg1 ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... );
    }
#line 1616
}; 
#line 1618
struct _Invoker_pmf_refwrap { 
#line 1619
static constexpr _Invoker_strategy _Strategy = _Invoker_strategy::_Pmf_refwrap; 
#line 1621
template < class _Decayed, class _Refwrap, class ... _Types2 >
    static constexpr auto _Call ( _Decayed _Pmf, _Refwrap _Rw, _Types2 && ... _Args2 )
        noexcept ( noexcept ( ( _Rw . get ( ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... ) ) )
        -> decltype ( ( _Rw . get ( ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... ) ) {
        return ( _Rw . get ( ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... );
    }
#line 1627
}; 
#line 1629
struct _Invoker_pmf_pointer { 
#line 1630
static constexpr _Invoker_strategy _Strategy = _Invoker_strategy::_Pmf_pointer; 
#line 1632
template < class _Decayed, class _Ty1, class ... _Types2 >
    static constexpr auto _Call ( _Decayed _Pmf, _Ty1 && _Arg1, _Types2 && ... _Args2 )
        noexcept ( noexcept ( ( ( * static_cast < _Ty1 && > ( _Arg1 ) ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... ) ) )
        -> decltype ( ( ( * static_cast < _Ty1 && > ( _Arg1 ) ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... ) ) {
        return ( ( * static_cast < _Ty1 && > ( _Arg1 ) ) .* _Pmf ) ( static_cast < _Types2 && > ( _Args2 ) ... );
    }
#line 1638
}; 
#line 1640
struct _Invoker_pmd_object { 
#line 1641
static constexpr _Invoker_strategy _Strategy = _Invoker_strategy::_Pmd_object; 
#line 1643
template < class _Decayed, class _Ty1 >
    static constexpr auto _Call ( _Decayed _Pmd, _Ty1 && _Arg1 ) noexcept -> decltype ( static_cast < _Ty1 && > ( _Arg1 ) .* _Pmd ) {
        return static_cast < _Ty1 && > ( _Arg1 ) .* _Pmd;
    }
#line 1647
}; 
#line 1649
struct _Invoker_pmd_refwrap { 
#line 1650
static constexpr _Invoker_strategy _Strategy = _Invoker_strategy::_Pmd_refwrap; 
#line 1652
template < class _Decayed, class _Refwrap >
    static constexpr auto _Call ( _Decayed _Pmd, _Refwrap _Rw ) noexcept -> decltype ( _Rw . get ( ) .* _Pmd ) {
        return _Rw . get ( ) .* _Pmd;
    }
#line 1656
}; 
#line 1658
struct _Invoker_pmd_pointer { 
#line 1659
static constexpr _Invoker_strategy _Strategy = _Invoker_strategy::_Pmd_pointer; 
#line 1661
template < class _Decayed, class _Ty1 >
    static constexpr auto _Call ( _Decayed _Pmd, _Ty1 && _Arg1 )
        noexcept ( noexcept ( ( * static_cast < _Ty1 && > ( _Arg1 ) ) .* _Pmd ) ) -> decltype ( ( * static_cast < _Ty1 && > ( _Arg1 ) ) .* _Pmd ) {
        return ( * static_cast < _Ty1 && > ( _Arg1 ) ) .* _Pmd;
    }
#line 1666
}; 
#line 1668
template< class _Callable, class _Ty1, class _Removed_cvref = _Remove_cvref_t< _Callable> , bool 
#line 1669
_Is_pmf = is_member_function_pointer_v< _Removed_cvref> , bool 
#line 1670
_Is_pmd = is_member_object_pointer_v< _Removed_cvref> > struct _Invoker1; 
#line 1673
template< class _Callable, class _Ty1, class _Removed_cvref> 
#line 1674
struct _Invoker1< _Callable, _Ty1, _Removed_cvref, true, false>  : public conditional_t< is_same_v< typename _Is_memfunptr< _Removed_cvref> ::_Class_type, _Remove_cvref_t< _Ty1> >  || is_base_of_v< typename _Is_memfunptr< _Removed_cvref> ::_Class_type, _Remove_cvref_t< _Ty1> > , _Invoker_pmf_object, conditional_t< _Is_specialization_v< _Remove_cvref_t< _Ty1> , reference_wrapper> , _Invoker_pmf_refwrap, _Invoker_pmf_pointer> >  { 
#line 1679
}; 
#line 1681
template< class _Callable, class _Ty1, class _Removed_cvref> 
#line 1682
struct _Invoker1< _Callable, _Ty1, _Removed_cvref, false, true>  : public conditional_t< is_same_v< typename _Is_member_object_pointer< _Removed_cvref> ::_Class_type, _Remove_cvref_t< _Ty1> >  || is_base_of_v< typename _Is_member_object_pointer< _Removed_cvref> ::_Class_type, _Remove_cvref_t< _Ty1> > , _Invoker_pmd_object, conditional_t< _Is_specialization_v< _Remove_cvref_t< _Ty1> , reference_wrapper> , _Invoker_pmd_refwrap, _Invoker_pmd_pointer> >  { 
#line 1688
}; 
#line 1690
template< class _Callable, class _Ty1, class _Removed_cvref> 
#line 1691
struct _Invoker1< _Callable, _Ty1, _Removed_cvref, false, false>  : public _Invoker_functor { }; 
#line 1693
template < class _Callable >
constexpr auto invoke ( _Callable && _Obj )
    noexcept ( noexcept ( static_cast < _Callable && > ( _Obj ) ( ) ) ) -> decltype ( static_cast < _Callable && > ( _Obj ) ( ) ) {
    return static_cast < _Callable && > ( _Obj ) ( );
}
#line 1699
template < class _Callable, class _Ty1, class ... _Types2 >
constexpr auto invoke ( _Callable && _Obj, _Ty1 && _Arg1, _Types2 && ... _Args2 )
    noexcept ( noexcept ( _Invoker1 < _Callable, _Ty1 > :: _Call (
        static_cast < _Callable && > ( _Obj ), static_cast < _Ty1 && > ( _Arg1 ), static_cast < _Types2 && > ( _Args2 ) ... ) ) )
    -> decltype ( _Invoker1 < _Callable, _Ty1 > :: _Call (
        static_cast < _Callable && > ( _Obj ), static_cast < _Ty1 && > ( _Arg1 ), static_cast < _Types2 && > ( _Args2 ) ... ) ) {
    if constexpr ( _Invoker1 < _Callable, _Ty1 > :: _Strategy == _Invoker_strategy :: _Functor ) {
        return static_cast < _Callable && > ( _Obj ) ( static_cast < _Ty1 && > ( _Arg1 ), static_cast < _Types2 && > ( _Args2 ) ... );
    } else if constexpr ( _Invoker1 < _Callable, _Ty1 > :: _Strategy == _Invoker_strategy :: _Pmf_object ) {
        return ( static_cast < _Ty1 && > ( _Arg1 ) .* _Obj ) ( static_cast < _Types2 && > ( _Args2 ) ... );
    } else if constexpr ( _Invoker1 < _Callable, _Ty1 > :: _Strategy == _Invoker_strategy :: _Pmf_refwrap ) {
        return ( _Arg1 . get ( ) .* _Obj ) ( static_cast < _Types2 && > ( _Args2 ) ... );
    } else if constexpr ( _Invoker1 < _Callable, _Ty1 > :: _Strategy == _Invoker_strategy :: _Pmf_pointer ) {
        return ( ( * static_cast < _Ty1 && > ( _Arg1 ) ) .* _Obj ) ( static_cast < _Types2 && > ( _Args2 ) ... );
    } else if constexpr ( _Invoker1 < _Callable, _Ty1 > :: _Strategy == _Invoker_strategy :: _Pmd_object ) {
        return static_cast < _Ty1 && > ( _Arg1 ) .* _Obj;
    } else if constexpr ( _Invoker1 < _Callable, _Ty1 > :: _Strategy == _Invoker_strategy :: _Pmd_refwrap ) {



        auto & _Ref = _Arg1 . get ( );
        return _Ref .* _Obj;

    } else {
        ;
        return ( * static_cast < _Ty1 && > ( _Arg1 ) ) .* _Obj;
    }
}
#line 1728 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
#pragma warning(push)
#pragma warning(disable : 4242)
#pragma warning(disable : 4244)
#pragma warning(disable : 4365)
#line 1733
template< class _From, class _To, bool  = is_convertible_v< _From, _To> , bool  = is_void_v< _To> > constexpr bool 
#line 1734
_Is_nothrow_convertible_v = noexcept(std::_Fake_copy_init< _To> (std::declval< _From> ())); 
#line 1736
#pragma warning(pop)
#line 1738
template< class _From, class _To, bool _IsVoid> constexpr bool 
#line 1739
_Is_nothrow_convertible_v< _From, _To, false, _IsVoid>  = false; 
#line 1741
template< class _From, class _To> constexpr bool 
#line 1742
_Is_nothrow_convertible_v< _From, _To, true, true>  = true; 
#line 1744
template< class _From, class _To> 
#line 1745
struct _Is_nothrow_convertible : public bool_constant< _Is_nothrow_convertible_v< _From, _To, is_convertible_v< _From, _To> , is_void_v< _To> > >  { 
#line 1747
}; 
#line 1757 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _From, class _To, class  = void> 
#line 1758
struct _Invoke_convertible : public false_type { }; 
#line 1760
template< class _From, class _To> 
#line 1761
struct _Invoke_convertible< _From, _To, void_t< decltype(std::_Fake_copy_init< _To> (std::_Returns_exactly< _From> ()))> >  : public true_type { 
#line 1762
}; 
#line 1764
template< class _From, class _To> 
#line 1765
struct _Invoke_nothrow_convertible : public bool_constant< noexcept(std::_Fake_copy_init< _To> (std::_Returns_exactly< _From> ()))>  { 
#line 1766
}; 
#line 1768
template< class _Result, bool _Nothrow> 
#line 1769
struct _Invoke_traits_common { 
#line 1770
using type = _Result; 
#line 1771
using _Is_invocable = true_type; 
#line 1772
using _Is_nothrow_invocable = bool_constant< _Nothrow> ; 
#line 1773
template< class _Rx> using _Is_invocable_r = bool_constant< disjunction_v< is_void< _Rx> , _Invoke_convertible< type, _Rx> > > ; 
#line 1775
template< class _Rx> using _Is_nothrow_invocable_r = bool_constant< conjunction_v< _Is_nothrow_invocable, disjunction< is_void< _Rx> , conjunction< _Invoke_convertible< type, _Rx> , _Invoke_nothrow_convertible< type, _Rx> > > > > ; 
#line 1779
}; 
#line 1781
template< class _Void, class _Callable> 
#line 1782
struct _Invoke_traits_zero { 
#line 1784
using _Is_invocable = false_type; 
#line 1785
using _Is_nothrow_invocable = false_type; 
#line 1786
template< class _Rx> using _Is_invocable_r = false_type; 
#line 1788
template< class _Rx> using _Is_nothrow_invocable_r = false_type; 
#line 1790
}; 
#line 1792
template< class _Callable> using _Decltype_invoke_zero = decltype(std::declval< _Callable> ()()); 
#line 1795
template< class _Callable> 
#line 1796
struct _Invoke_traits_zero< void_t< _Decltype_invoke_zero< _Callable> > , _Callable>  : public _Invoke_traits_common< _Decltype_invoke_zero< _Callable> , noexcept(std::declval< _Callable> ()())>  { 
#line 1797
}; 
#line 1799
template< class _Void, class ..._Types> 
#line 1800
struct _Invoke_traits_nonzero { 
#line 1802
using _Is_invocable = false_type; 
#line 1803
using _Is_nothrow_invocable = false_type; 
#line 1804
template< class _Rx> using _Is_invocable_r = false_type; 
#line 1806
template< class _Rx> using _Is_nothrow_invocable_r = false_type; 
#line 1808
}; 
#line 1810
template< class _Callable, class _Ty1, class ..._Types2> using _Decltype_invoke_nonzero = decltype(_Invoker1< _Callable, _Ty1> ::_Call(std::declval< _Callable> (), std::declval< _Ty1> (), std::declval< _Types2> ()...)); 
#line 1814
template< class _Callable, class _Ty1, class ..._Types2> 
#line 1815
struct _Invoke_traits_nonzero< void_t< _Decltype_invoke_nonzero< _Callable, _Ty1, _Types2...> > , _Callable, _Ty1, _Types2...>  : public _Invoke_traits_common< _Decltype_invoke_nonzero< _Callable, _Ty1, _Types2...> , noexcept(_Invoker1< _Callable, _Ty1> ::_Call(std::declval< _Callable> (), std::declval< _Ty1> (), std::declval< _Types2> ()...))>  { 
#line 1818
}; 
#line 1820
template< class _Callable, class ..._Args> using _Select_invoke_traits = conditional_t< sizeof...(_Args) == (0), _Invoke_traits_zero< void, _Callable> , _Invoke_traits_nonzero< void, _Callable, _Args...> > ; 
#line 1825
template< class _Fty> 
#line 1826
struct result_of { 
#line 1827
static_assert((_Always_false< _Fty> ), "result_of<CallableType> is invalid; use result_of<CallableType(zero or more argument types)> instead.");
#line 1829
}; 
#line 1837
template< class _Callable, class ..._Args> struct result_of< _Callable __cdecl (_Args ...)>  : public _Select_invoke_traits< _Callable, _Args...>  { }; template< class _Callable, class ..._Args> struct result_of< _Callable __vectorcall (_Args ...)>  : public _Select_invoke_traits< _Callable, _Args...>  { }; 
#line 1840
__pragma( warning(push)) __pragma( warning(disable : 4996)) 
#line 1841
template< class _Ty> using result_of_t = typename result_of< _Ty> ::type; 
#line 1843
__pragma( warning(pop)) 
#line 1846 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Callable, class ..._Args> using _Invoke_result_t = typename _Select_invoke_traits< _Callable, _Args...> ::type; 
#line 1849
template< class _Rx, class _Callable, class ..._Args> using _Is_invocable_r_ = typename conditional< sizeof...(_Args) == (0), _Invoke_traits_zero< void, _Callable> , _Invoke_traits_nonzero< void, _Callable, _Args...> > ::type::template _Is_invocable_r< _Rx> ; 
#line 1852
template< class _Rx, class _Callable, class ..._Args> 
#line 1853
struct _Is_invocable_r : public _Is_invocable_r_< _Rx, _Callable, _Args...>  { 
#line 1855
}; 
#line 1926 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> 
#line 1927
struct _Function_args { }; 
#line 1935
template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...)>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...)>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) const>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) const>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) volatile>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) volatile>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) const volatile>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) const volatile>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) &>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) &>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) const &>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) const &>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) volatile &>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) volatile &>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) const volatile &>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) const volatile &>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) &&>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) &&>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) const &&>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) const &&>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) volatile &&>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) volatile &&>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __cdecl (_Types ...) const volatile &&>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret __vectorcall (_Types ...) const volatile &&>  : public _Arg_types< _Types...>  { using result_type = _Ret; }; 
#line 1944
template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...)>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) const>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) volatile>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) const volatile>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) &>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) const &>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) volatile &>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) const volatile &>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) &&>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) const &&>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) volatile &&>  { using result_type = _Ret; }; template< class _Ret, class ..._Types> struct _Function_args< _Ret (_Types ..., ...) const volatile &&>  { using result_type = _Ret; }; 
#line 1947
template< class _Ty, class  = void> 
#line 1948
struct _Weak_result_type { }; 
#line 1950
__pragma( warning(push)) __pragma( warning(disable : 4996)) 
#line 1951
template< class _Ty> 
#line 1952
struct _Weak_result_type< _Ty, void_t< typename _Ty::result_type> >  { 
#line 1953
using result_type = typename _Ty::result_type; 
#line 1954
}; 
#line 1955
__pragma( warning(pop)) 
#line 1957
template< class _Ty, class  = void> 
#line 1958
struct _Weak_argument_type : public _Weak_result_type< _Ty>  { }; 
#line 1960
__pragma( warning(push)) __pragma( warning(disable : 4996)) 
#line 1961
template< class _Ty> 
#line 1962
struct _Weak_argument_type< _Ty, void_t< typename _Ty::argument_type> >  : public _Weak_result_type< _Ty>  { 
#line 1964
using argument_type = typename _Ty::argument_type; 
#line 1965
}; 
#line 1966
__pragma( warning(pop)) 
#line 1968
template< class _Ty, class  = void> 
#line 1969
struct _Weak_binary_args : public _Weak_argument_type< _Ty>  { }; 
#line 1971
__pragma( warning(push)) __pragma( warning(disable : 4996)) 
#line 1972
template< class _Ty> 
#line 1973
struct _Weak_binary_args< _Ty, void_t< typename _Ty::first_argument_type, typename _Ty::second_argument_type> >  : public _Weak_argument_type< _Ty>  { 
#line 1976
using first_argument_type = typename _Ty::first_argument_type; 
#line 1977
using second_argument_type = typename _Ty::second_argument_type; 
#line 1978
}; 
#line 1979
__pragma( warning(pop)) 
#line 1981
template< class _Ty> using _Weak_types = conditional_t< is_function_v< remove_pointer_t< _Ty> > , _Function_args< remove_pointer_t< _Ty> > , conditional_t< is_member_function_pointer_v< _Ty> , _Is_memfunptr< remove_cv_t< _Ty> > , _Weak_binary_args< _Ty> > > ; 
#line 1985
template< class _Ty> void _Refwrap_ctor_fun(_Identity_t< _Ty &> ) noexcept; 
#line 1987
template < class _Ty >
void _Refwrap_ctor_fun ( _Identity_t < _Ty && > ) = delete;
#line 1990
template< class _Ty, class _Uty, class  = void> 
#line 1991
struct _Refwrap_has_ctor_from : public false_type { }; 
#line 1993
template< class _Ty, class _Uty> 
#line 1994
struct _Refwrap_has_ctor_from< _Ty, _Uty, void_t< decltype(std::_Refwrap_ctor_fun< _Ty> (std::declval< _Uty> ()))> >  : public true_type { 
#line 1995
}; 
#line 1997
template < class _Ty >
class reference_wrapper

    : public _Weak_types < _Ty >

{
public :
    static_assert ( is_object_v < _Ty > || is_function_v < _Ty >,
        "reference_wrapper<T> requires T to be an object type or a function type." );

    using type = _Ty;

    template < class _Uty, enable_if_t < conjunction_v < negation < is_same < _Remove_cvref_t < _Uty >, reference_wrapper >>,
                                          _Refwrap_has_ctor_from < _Ty, _Uty >>,
                              int > = 0 >
    inline reference_wrapper ( _Uty && _Val ) noexcept (
        noexcept ( :: std :: _Refwrap_ctor_fun < _Ty > ( :: std :: declval < _Uty > ( ) ) ) ) {
        _Ty & _Ref = static_cast < _Uty && > ( _Val );
        _Ptr = :: std :: addressof ( _Ref );
    }

    inline operator _Ty & ( ) const noexcept {
        return * _Ptr;
    }

    [ [ nodiscard ] ] inline _Ty & get ( ) const noexcept {
        return * _Ptr;
    }

private :
    _Ty * _Ptr { };

public :
    template < class ... _Types >
    inline auto operator ( ) ( _Types && ... _Args ) const
        noexcept ( noexcept ( :: std :: invoke ( * _Ptr, static_cast < _Types && > ( _Args ) ... ) ) )
        -> decltype ( :: std :: invoke ( * _Ptr, static_cast < _Types && > ( _Args ) ... ) ) {
        return :: std :: invoke ( * _Ptr, static_cast < _Types && > ( _Args ) ... );
    }
};
#line 2043 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template < class _Ty >
[ [ nodiscard ] ] inline reference_wrapper < _Ty > ref ( _Ty & _Val ) noexcept {
    return reference_wrapper < _Ty > ( _Val );
}
#line 2048
template < class _Ty >
void ref ( const _Ty && ) = delete;
#line 2051
template < class _Ty >
[ [ nodiscard ] ] inline reference_wrapper < _Ty > ref ( reference_wrapper < _Ty > _Val ) noexcept {
    return _Val;
}
#line 2056
template < class _Ty >
[ [ nodiscard ] ] inline reference_wrapper < const _Ty > cref ( const _Ty & _Val ) noexcept {
    return reference_wrapper < const _Ty > ( _Val );
}
#line 2061
template < class _Ty >
void cref ( const _Ty && ) = delete;
#line 2064
template < class _Ty >
[ [ nodiscard ] ] inline reference_wrapper < const _Ty > cref ( reference_wrapper < _Ty > _Val ) noexcept {
    return _Val;
}
#line 2089 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class _Ty> struct _Is_swappable; 
#line 2092
template< class _Ty> struct _Is_nothrow_swappable; 
#line 2098 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template < class _Ty, int _Enabled = 0 >

inline void swap ( _Ty &, _Ty & ) noexcept ( is_nothrow_move_constructible_v < _Ty > && is_nothrow_move_assignable_v < _Ty > );
#line 2102 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template < class _Ty, size_t _Size, enable_if_t < _Is_swappable < _Ty > :: value, int > = 0 >
inline void swap ( _Ty ( & ) [ _Size ], _Ty ( & ) [ _Size ] ) noexcept ( _Is_nothrow_swappable < _Ty > :: value );
#line 2105
template< class _Ty1, class _Ty2, class  = void> 
#line 2106
struct _Swappable_with_helper : public false_type { }; 
#line 2108
template< class _Ty1, class _Ty2> 
#line 2109
struct _Swappable_with_helper< _Ty1, _Ty2, void_t< decltype(swap(std::declval< _Ty1> (), std::declval< _Ty2> ()))> >  : public true_type { 
#line 2110
}; 
#line 2112
template< class _Ty1, class _Ty2> 
#line 2113
struct _Is_swappable_with : public bool_constant< conjunction_v< _Swappable_with_helper< _Ty1, _Ty2> , _Swappable_with_helper< _Ty2, _Ty1> > >  { 
#line 2116
}; 
#line 2118
template< class _Ty> 
#line 2119
struct _Is_swappable : public _Is_swappable_with< add_lvalue_reference_t< _Ty> , add_lvalue_reference_t< _Ty> > ::type { 
#line 2121
}; 
#line 2123
template< class _Ty1, class _Ty2> 
#line 2124
struct _Swap_cannot_throw : public bool_constant< noexcept(swap(std::declval< _Ty1> (), std::declval< _Ty2> ())) && noexcept(swap(std::declval< _Ty2> (), std::declval< _Ty1> ()))>  { 
#line 2128
}; 
#line 2130
template< class _Ty1, class _Ty2> 
#line 2131
struct _Is_nothrow_swappable_with : public bool_constant< conjunction_v< _Is_swappable_with< _Ty1, _Ty2> , _Swap_cannot_throw< _Ty1, _Ty2> > >  { 
#line 2135
}; 
#line 2137
template< class _Ty> 
#line 2138
struct _Is_nothrow_swappable : public _Is_nothrow_swappable_with< add_lvalue_reference_t< _Ty> , add_lvalue_reference_t< _Ty> > ::type { 
#line 2141
}; 
#line 2178 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
namespace _Has_ADL_swap_detail { 
#line 2182 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
void swap(); 
#line 2185 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template< class , class  = void> 
#line 2186
struct _Has_ADL_swap : public false_type { }; 
#line 2187
template< class _Ty> 
#line 2188
struct _Has_ADL_swap< _Ty, void_t< decltype(swap(std::declval< _Ty &> (), std::declval< _Ty &> ()))> >  : public true_type { }; 
#line 2189
}
#line 2190
using _Has_ADL_swap_detail::_Has_ADL_swap;
#line 2192
template< class _Ty> constexpr bool 
#line 2193
_Is_trivially_swappable_v = conjunction_v< is_trivially_destructible< _Ty> , is_trivially_move_constructible< _Ty> , is_trivially_move_assignable< _Ty> , negation< _Has_ADL_swap_detail::_Has_ADL_swap< _Ty> > > ; 
#line 2201
template< class _Ty> 
#line 2202
struct _Is_trivially_swappable : public bool_constant< _Is_trivially_swappable_v< _Ty> >  { 
#line 2204
}; 
#line 2269 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
template < class _Bitmask >
[ [ nodiscard ] ] constexpr bool _Bitmask_includes_any ( _Bitmask _Left, _Bitmask _Elements ) noexcept {
    return ( _Left & _Elements ) != _Bitmask { };
}
#line 2274
template < class _Bitmask >
[ [ nodiscard ] ] constexpr bool _Bitmask_includes_all ( _Bitmask _Left, _Bitmask _Elements ) noexcept {
    return ( _Left & _Elements ) == _Elements;
}
#line 2282
constexpr size_t _FNV_offset_basis = 14695981039346656037Ui64; 
#line 2283
constexpr size_t _FNV_prime = 1099511628211Ui64; 
#line 2289 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
[[nodiscard]] inline size_t _Fnv1a_append_bytes(size_t _Val, const unsigned char *const _First, const size_t 
#line 2290
_Count) noexcept { 
#line 2291
for (size_t _Idx = (0); _Idx < _Count; ++_Idx) { 
#line 2292
_Val ^= (static_cast< size_t>(_First[_Idx])); 
#line 2293
_Val *= _FNV_prime; 
#line 2294
}  
#line 2296
return _Val; 
#line 2297
} 
#line 2299
template < class _Ty >
[ [ nodiscard ] ] size_t _Fnv1a_append_range ( const size_t _Val, const _Ty * const _First,
    const _Ty * const _Last ) noexcept {
    static_assert ( is_trivial_v < _Ty >, "Only trivial types can be directly hashed." );
    const auto _Firstb = reinterpret_cast < const unsigned char * > ( _First );
    const auto _Lastb = reinterpret_cast < const unsigned char * > ( _Last );
    return _Fnv1a_append_bytes ( _Val, _Firstb, static_cast < size_t > ( _Lastb - _Firstb ) );
}
#line 2308
template < class _Kty >
[ [ nodiscard ] ] size_t _Fnv1a_append_value (
    const size_t _Val, const _Kty & _Keyval ) noexcept {
    static_assert ( is_trivial_v < _Kty >, "Only trivial types can be directly hashed." );
    return _Fnv1a_append_bytes ( _Val, & reinterpret_cast < const unsigned char & > ( _Keyval ), sizeof ( _Kty ) );
}
#line 2315
template < class _Kty >
[ [ nodiscard ] ] size_t _Hash_representation ( const _Kty & _Keyval ) noexcept {
    return _Fnv1a_append_value ( _FNV_offset_basis, _Keyval );
}
#line 2320
template < class _Kty >
[ [ nodiscard ] ] size_t _Hash_array_representation (
    const _Kty * const _First, const size_t _Count ) noexcept {
    static_assert ( is_trivial_v < _Kty >, "Only trivial types can be directly hashed." );
    return _Fnv1a_append_bytes (
        _FNV_offset_basis, reinterpret_cast < const unsigned char * > ( _First ), _Count * sizeof ( _Kty ) );
}
#line 2328
template < class _Kty >
struct hash;
#line 2331
template < class _Kty, bool _Enabled >
struct _Conditionally_enabled_hash {
    using argument_type = _Kty;
    using result_type = size_t;

    [ [ nodiscard ] ] size_t operator ( ) ( const _Kty & _Keyval ) const
        noexcept ( noexcept ( hash < _Kty > :: _Do_hash ( _Keyval ) ) ) {
        return hash < _Kty > :: _Do_hash ( _Keyval );
    }
};
#line 2342
template< class _Kty> 
#line 2343
struct _Conditionally_enabled_hash< _Kty, false>  { 
#line 2344
_Conditionally_enabled_hash() = delete;
#line 2345
_Conditionally_enabled_hash(const _Conditionally_enabled_hash &) = delete;
#line 2346
_Conditionally_enabled_hash(_Conditionally_enabled_hash &&) = delete;
#line 2347
_Conditionally_enabled_hash &operator=(const _Conditionally_enabled_hash &) = delete;
#line 2348
_Conditionally_enabled_hash &operator=(_Conditionally_enabled_hash &&) = delete;
#line 2349
}; 
#line 2351
template < class _Kty >
struct hash : _Conditionally_enabled_hash < _Kty, ! is_const_v < _Kty > && ! is_volatile_v < _Kty >
                                                    && ( is_enum_v < _Kty > || is_integral_v < _Kty > || is_pointer_v < _Kty > ) > {

    static size_t _Do_hash ( const _Kty & _Keyval ) noexcept {
        return _Hash_representation ( _Keyval );
    }
};
#line 2361
#pragma pack(8)
template<> 
#line 2361
struct hash< float>  { 
#line 2362
using argument_type = float; 
#line 2363
using result_type = size_t; 
#line 2364
[[nodiscard]] size_t operator()(const float _Keyval) const noexcept { 
#line 2365
return _Hash_representation((_Keyval == (0.0F)) ? (0.0F) : _Keyval); 
#line 2366
} 
#line 2367
}; 
#pragma pack()
#line 2370
#pragma pack(8)
template<> 
#line 2370
struct hash< double>  { 
#line 2371
using argument_type = double; 
#line 2372
using result_type = size_t; 
#line 2373
[[nodiscard]] size_t operator()(const double _Keyval) const noexcept { 
#line 2374
return _Hash_representation((_Keyval == (0.0)) ? (0.0) : _Keyval); 
#line 2375
} 
#line 2376
}; 
#pragma pack()
#line 2379
#pragma pack(8)
template<> 
#line 2379
struct hash< long double>  { 
#line 2380
using argument_type = long double; 
#line 2381
using result_type = size_t; 
#line 2382
[[nodiscard]] size_t operator()(const long double _Keyval) const noexcept { 
#line 2383
return _Hash_representation((_Keyval == (0.0L)) ? (0.0L) : _Keyval); 
#line 2384
} 
#line 2385
}; 
#pragma pack()
#line 2388
#pragma pack(8)
template<> 
#line 2388
struct hash< std::nullptr_t>  { 
#line 2389
using argument_type = nullptr_t; 
#line 2390
using result_type = size_t; 
#line 2391
[[nodiscard]] size_t operator()(nullptr_t) const noexcept { 
#line 2392
void *_Null{((void *)0i64)}; 
#line 2393
return _Hash_representation(_Null); 
#line 2394
} 
#line 2395
}; 
#pragma pack()
template< class _Kty, class  = void> 
#pragma pack(8)
#line 2398
struct _Is_nothrow_hashable : public false_type { }; 
#pragma pack()
template< class _Kty> 
#pragma pack(8)
#line 2401
struct _Is_nothrow_hashable< _Kty, void_t< decltype(hash< _Kty> {}(std::declval< const _Kty &> ()))> >  : public bool_constant< noexcept(hash< _Kty> {}(std::declval< const _Kty &> ()))>  { 
#line 2402
}; 
#pragma pack()
template < class _Ty = void >
struct less {
    using first_argument_type = _Ty;
    using second_argument_type = _Ty;
    using result_type = bool;

    [ [ nodiscard ] ] constexpr bool operator ( ) ( const _Ty & _Left, const _Ty & _Right ) const
        noexcept ( noexcept ( :: std :: _Fake_copy_init < bool > ( _Left < _Right ) ) ) {
        return _Left < _Right;
    }
};
#line 2417
template<> struct less< void>  { 
#line 2418
template < class _Ty1, class _Ty2 >
    [ [ nodiscard ] ] constexpr auto operator ( ) ( _Ty1 && _Left, _Ty2 && _Right ) const
        noexcept ( noexcept ( static_cast < _Ty1 && > ( _Left ) < static_cast < _Ty2 && > ( _Right ) ) )
        -> decltype ( static_cast < _Ty1 && > ( _Left ) < static_cast < _Ty2 && > ( _Right ) ) {
        return static_cast < _Ty1 && > ( _Left ) < static_cast < _Ty2 && > ( _Right );
    }
#line 2425
using is_transparent = int; 
#line 2426
}; 
#line 2430
template< class _FloatingType> struct _Floating_type_traits; 
#line 2434
template<> struct _Floating_type_traits< float>  { 
#line 2435
static constexpr int32_t _Mantissa_bits = 24; 
#line 2436
static constexpr int32_t _Exponent_bits = 8; 
#line 2437
static constexpr int32_t _Maximum_binary_exponent = 127; 
#line 2438
static constexpr int32_t _Minimum_binary_exponent = (-126); 
#line 2439
static constexpr int32_t _Exponent_bias = 127; 
#line 2440
static constexpr int32_t _Sign_shift = 31; 
#line 2441
static constexpr int32_t _Exponent_shift = 23; 
#line 2443
using _Uint_type = uint32_t; 
#line 2445
static constexpr uint32_t _Exponent_mask = 255U; 
#line 2446
static constexpr uint32_t _Normal_mantissa_mask = 16777215U; 
#line 2447
static constexpr uint32_t _Denormal_mantissa_mask = 8388607U; 
#line 2448
static constexpr uint32_t _Special_nan_mantissa_mask = 4194304U; 
#line 2449
static constexpr uint32_t _Shifted_sign_mask = 2147483648U; 
#line 2450
static constexpr uint32_t _Shifted_exponent_mask = 2139095040U; 
#line 2452
static constexpr float _Minimum_value = (1.175494351e-38F); 
#line 2453
static constexpr float _Maximum_value = (3.402823466e+38F); 
#line 2454
}; 
#line 2457
template<> struct _Floating_type_traits< double>  { 
#line 2458
static constexpr int32_t _Mantissa_bits = 53; 
#line 2459
static constexpr int32_t _Exponent_bits = 11; 
#line 2460
static constexpr int32_t _Maximum_binary_exponent = 1023; 
#line 2461
static constexpr int32_t _Minimum_binary_exponent = (-1022); 
#line 2462
static constexpr int32_t _Exponent_bias = 1023; 
#line 2463
static constexpr int32_t _Sign_shift = 63; 
#line 2464
static constexpr int32_t _Exponent_shift = 52; 
#line 2466
using _Uint_type = uint64_t; 
#line 2468
static constexpr uint64_t _Exponent_mask = (2047U); 
#line 2469
static constexpr uint64_t _Normal_mantissa_mask = 9007199254740991Ui64; 
#line 2470
static constexpr uint64_t _Denormal_mantissa_mask = 4503599627370495Ui64; 
#line 2471
static constexpr uint64_t _Special_nan_mantissa_mask = 2251799813685248Ui64; 
#line 2472
static constexpr uint64_t _Shifted_sign_mask = 9223372036854775808Ui64; 
#line 2473
static constexpr uint64_t _Shifted_exponent_mask = 9218868437227405312Ui64; 
#line 2475
static constexpr double _Minimum_value = (2.225073858507201383e-308); 
#line 2476
static constexpr double _Maximum_value = (1.797693134862315708e+308); 
#line 2477
}; 
#line 2480
template<> struct _Floating_type_traits< long double>  : public std::_Floating_type_traits< double>  { }; 
#line 2484
template < class _To, class _From,
    enable_if_t < conjunction_v < bool_constant < sizeof ( _To ) == sizeof ( _From ) >, is_trivially_copyable < _To >,
                    is_trivially_copyable < _From > >,
        int > = 0 >
[ [ nodiscard ] ] constexpr _To _Bit_cast ( const _From & _Val ) noexcept {
    return __builtin_bit_cast ( _To, _Val );
}
#line 2492
template < class _Ty, enable_if_t < is_floating_point_v < _Ty >, int > = 0 >
[ [ nodiscard ] ] constexpr auto _Float_abs_bits ( const _Ty & _Xx ) noexcept {
    using _Traits = _Floating_type_traits < _Ty >;
    using _Uint_type = typename _Traits :: _Uint_type;
    const auto _Bits = _Bit_cast < _Uint_type > ( _Xx );
    return _Bits & ~ _Traits :: _Shifted_sign_mask;
}
#line 2500
template < class _Ty, enable_if_t < is_floating_point_v < _Ty >, int > = 0 >
[ [ nodiscard ] ] constexpr _Ty _Float_abs ( const _Ty _Xx ) noexcept {
    return _Bit_cast < _Ty > ( _Float_abs_bits ( _Xx ) );
}
#line 2505
template < class _Ty, enable_if_t < is_floating_point_v < _Ty >, int > = 0 >
[ [ nodiscard ] ] constexpr bool _Is_nan ( const _Ty _Xx ) noexcept {
    using _Traits = _Floating_type_traits < _Ty >;
    return _Float_abs_bits ( _Xx ) > _Traits :: _Shifted_exponent_mask;
}
#line 2511
template < class _Ty, enable_if_t < is_floating_point_v < _Ty >, int > = 0 >
[ [ nodiscard ] ] constexpr bool _Is_finite ( const _Ty _Xx ) noexcept {
    using _Traits = _Floating_type_traits < _Ty >;
    return _Float_abs_bits ( _Xx ) < _Traits :: _Shifted_exponent_mask;
}
#line 2517
template< bool _IsConst, class _Ty> using _Maybe_const = conditional_t< _IsConst, const _Ty, _Ty> ; 
#line 2520
template < class _Ty, size_t _Size >
void _Swap_trivial_arrays ( _Ty ( & _Left ) [ _Size ], _Ty ( & _Right ) [ _Size ] ) noexcept {






    auto _Left_ptr = reinterpret_cast < unsigned char * > ( & _Left );
    auto _Right_ptr = reinterpret_cast < unsigned char * > ( & _Right );

    constexpr size_t _Part_size_bytes = 64;
    constexpr size_t _Size_bytes = _Size * sizeof ( _Ty );
    constexpr size_t _Size_tail = _Size_bytes % _Part_size_bytes;
    constexpr size_t _Size_parts = _Size_bytes - _Size_tail;

    if constexpr ( _Size_parts != 0 ) {
        const auto _Stop = _Left_ptr + _Size_parts;
        do {
            unsigned char _Buf [ _Part_size_bytes ];
            :: memcpy ( _Buf, _Left_ptr, _Part_size_bytes );
            :: memcpy ( _Left_ptr, _Right_ptr, _Part_size_bytes );
            :: memcpy ( _Right_ptr, _Buf, _Part_size_bytes );
            _Left_ptr += _Part_size_bytes;
            _Right_ptr += _Part_size_bytes;

        } while ( _Left_ptr != _Stop );
    }

    if constexpr ( _Size_tail != 0 ) {
        unsigned char _Buf [ _Size_tail ];
        :: memcpy ( _Buf, _Left_ptr, _Size_tail );
        :: memcpy ( _Left_ptr, _Right_ptr, _Size_tail );
        :: memcpy ( _Right_ptr, _Buf, _Size_tail );
    }
}
#line 2558
__pragma( warning(push)) __pragma( warning(disable : 4996)) 
#line 2559
namespace [[deprecated("warning STL4002: The non-Standard std::tr1 namespace and TR1-only machinery are deprecated and will be REMOVED. You can define _" "SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING to suppress this warning.")]] tr1 { 
#line 2560
using std::add_const;
#line 2561
using std::add_cv;
#line 2562
using std::add_pointer;
#line 2563
using std::add_volatile;
#line 2564
using std::aligned_storage;
#line 2565
using std::alignment_of;
#line 2566
using std::conditional;
#line 2567
using std::decay;
#line 2568
using std::enable_if;
#line 2569
using std::extent;
#line 2570
using std::false_type;
#line 2571
using std::has_virtual_destructor;
#line 2572
using std::integral_constant;
#line 2573
using std::is_abstract;
#line 2574
using std::is_arithmetic;
#line 2575
using std::is_array;
#line 2576
using std::is_base_of;
#line 2577
using std::is_class;
#line 2578
using std::is_compound;
#line 2579
using std::is_const;
#line 2580
using std::is_convertible;
#line 2581
using std::is_empty;
#line 2582
using std::is_enum;
#line 2583
using std::is_floating_point;
#line 2584
using std::is_function;
#line 2585
using std::is_fundamental;
#line 2586
using std::is_integral;
#line 2587
using std::is_member_function_pointer;
#line 2588
using std::is_member_object_pointer;
#line 2589
using std::is_member_pointer;
#line 2590
using std::is_object;
#line 2591
using std::is_pod;
#line 2592
using std::is_pointer;
#line 2593
using std::is_polymorphic;
#line 2594
using std::is_reference;
#line 2595
using std::is_same;
#line 2596
using std::is_scalar;
#line 2597
using std::is_signed;
#line 2598
using std::is_union;
#line 2599
using std::is_unsigned;
#line 2600
using std::is_void;
#line 2601
using std::is_volatile;
#line 2602
using std::make_signed;
#line 2603
using std::make_unsigned;
#line 2604
using std::rank;
#line 2605
using std::remove_all_extents;
#line 2606
using std::remove_const;
#line 2607
using std::remove_cv;
#line 2608
using std::remove_extent;
#line 2609
using std::remove_pointer;
#line 2610
using std::remove_reference;
#line 2611
using std::remove_volatile;
#line 2612
using std::true_type;
#line 2613
using std::cref;
#line 2614
using std::ref;
#line 2615
using std::reference_wrapper;
#line 2617
using std::result_of;
#line 2619 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
using std::hash;
#line 2620
}
#line 2621
__pragma( warning(pop)) 
#line 2624 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\type_traits"
}
#line 2633
#pragma warning(pop)
#pragma pack ( pop )
#line 22 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4324 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4868 4988 5026 5027 5045 5220 5246 6294 4984 5053 )
#line 39
namespace std { 
#line 40
template < class _Ty, _Ty ... _Vals >
struct integer_sequence {
    static_assert ( is_integral_v < _Ty >, "integer_sequence<T, I...> requires T to be an integral type." );

    using value_type = _Ty;

    [ [ nodiscard ] ] static constexpr size_t size ( ) noexcept {
        return sizeof ... ( _Vals );
    }
};
#line 51
template< class _Ty, _Ty _Size> using make_integer_sequence = __make_integer_seq< integer_sequence, _Ty, _Size> ; 
#line 54
template< size_t ..._Vals> using index_sequence = integer_sequence< unsigned __int64, _Vals...> ; 
#line 57
template< size_t _Size> using make_index_sequence = make_integer_sequence< unsigned __int64, _Size> ; 
#line 60
template< class ..._Types> using index_sequence_for = make_index_sequence< sizeof...(_Types)> ; 
#line 63
template < class _Ty, class _Pr >
[ [ nodiscard ] ] constexpr const _Ty & ( max ) ( const _Ty & _Left, const _Ty & _Right,
    _Pr _Pred ) noexcept ( noexcept ( _Pred ( _Left, _Right ) ) ) {

    return _Pred ( _Left, _Right ) ? _Right : _Left;
}
#line 70
#pragma warning(push)
#pragma warning(disable : 28285)
template < class _Ty >
[ [ nodiscard ] ] constexpr const _Ty &
    ( max ) ( const _Ty & _Left, const _Ty & _Right )
        noexcept ( noexcept ( _Left < _Right ) ) {

    return _Left < _Right ? _Right : _Left;
}
#pragma warning(pop)
#line 81
template< class _Ty, class _Pr> 
#line 82
[[nodiscard]] constexpr _Ty 
#line 81
max(initializer_list< _Ty> , _Pr); 
#line 84
template< class _Ty> 
#line 85
[[nodiscard]] constexpr _Ty 
#line 84
max(initializer_list< _Ty> ); 
#line 87
template < class _Ty, class _Pr >
[ [ nodiscard ] ] constexpr const _Ty & ( min ) ( const _Ty & _Left, const _Ty & _Right,
    _Pr _Pred ) noexcept ( noexcept ( _Pred ( _Right, _Left ) ) ) {

    return _Pred ( _Right, _Left ) ? _Right : _Left;
}
#line 94
#pragma warning(push)
#pragma warning(disable : 28285)
template < class _Ty >
[ [ nodiscard ] ] constexpr const _Ty &
    ( min ) ( const _Ty & _Left, const _Ty & _Right )
        noexcept ( noexcept ( _Right < _Left ) ) {

    return _Right < _Left ? _Right : _Left;
}
#pragma warning(pop)
#line 105
template< class _Ty, class _Pr> 
#line 106
[[nodiscard]] constexpr _Ty 
#line 105
min(initializer_list< _Ty> , _Pr); 
#line 108
template< class _Ty> 
#line 109
[[nodiscard]] constexpr _Ty 
#line 108
min(initializer_list< _Ty> ); 
#line 111
template < class _Ty, size_t _Size, enable_if_t < _Is_swappable < _Ty > :: value, int > >
inline void swap ( _Ty ( & _Left ) [ _Size ], _Ty ( & _Right ) [ _Size ] ) noexcept ( _Is_nothrow_swappable < _Ty > :: value ) {
    if ( & _Left == & _Right ) {
        return;
    }

    if constexpr ( _Is_trivially_swappable_v < _Ty > ) {
        if ( ! :: std :: _Is_constant_evaluated ( ) ) {
            :: std :: _Swap_trivial_arrays ( _Left, _Right );
            return;
        }
    }

    _Ty * _First1 = _Left;
    _Ty * _Last1 = _First1 + _Size;
    _Ty * _First2 = _Right;
    for (; _First1 != _Last1; ++ _First1, ++ _First2 ) {
        swap ( * _First1, * _First2 );
    }
}
#line 135 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
template < class _Ty, int _Enabled >

inline void swap ( _Ty & _Left, _Ty & _Right )
    noexcept ( is_nothrow_move_constructible_v < _Ty > && is_nothrow_move_assignable_v < _Ty > ) {
    _Ty _Tmp = :: std :: move ( _Left );
    _Left = :: std :: move ( _Right );
    _Right = :: std :: move ( _Tmp );
}
#line 144 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
#pragma pack(8)
#line 144
struct piecewise_construct_t { 
#line 145
explicit piecewise_construct_t() = default;
#line 146
}; 
#pragma pack()
constexpr piecewise_construct_t piecewise_construct{}; 
#line 150
#pragma pack(8)
#line 150
struct _Ignore { 
#line 151
template < class _Ty >
    constexpr const _Ignore & operator = ( const _Ty & ) const noexcept {

        return * this;
    }
#line 156
}; 
#pragma pack(8)
constexpr _Ignore ignore{}; 
#line 160
template< class ..._Types> class tuple; 
#line 163
template < class _Ty1, class _Ty2 >
struct pair;
#line 166
template< class _Ty, size_t _Size> class array; 
#line 169
template< class _Tuple> struct tuple_size; 
#line 172
template< class _Ty> constexpr size_t 
#line 173
tuple_size_v = (tuple_size< _Ty> ::value); 
#line 175
template< size_t _Index, class _Tuple> struct tuple_element; 
#line 178
template< size_t _Index, class _Tuple> using tuple_element_t = typename tuple_element< _Index, _Tuple> ::type; 
#line 181
template< size_t _Index, class ..._Types> 
#line 182
[[nodiscard]] constexpr auto &&
#line 181
_Tuple_get(tuple< _Types...>  && _Tuple) noexcept; 
#line 184
template< size_t _Index, class ..._Types> 
#line 185
[[nodiscard]] constexpr tuple_element_t< _Index, tuple< _Types...> >  &
#line 184
get(tuple< _Types...>  & _Tuple) noexcept; 
#line 187
template< size_t _Index, class ..._Types> 
#line 188
[[nodiscard]] constexpr const tuple_element_t< _Index, tuple< _Types...> >  &
#line 187
get(const tuple< _Types...>  & _Tuple) noexcept; 
#line 190
template< size_t _Index, class ..._Types> 
#line 191
[[nodiscard]] constexpr tuple_element_t< _Index, tuple< _Types...> >  &&
#line 190
get(tuple< _Types...>  && _Tuple) noexcept; 
#line 193
template< size_t _Index, class ..._Types> 
#line 194
[[nodiscard]] constexpr const tuple_element_t< _Index, tuple< _Types...> >  &&
#line 193
get(const tuple< _Types...>  && _Tuple) noexcept; 
#line 196
template< size_t _Idx, class _Ty, size_t _Size> 
#line 197
[[nodiscard]] constexpr _Ty &
#line 196
get(array< _Ty, _Size>  & _Arr) noexcept; 
#line 199
template< size_t _Idx, class _Ty, size_t _Size> 
#line 200
[[nodiscard]] constexpr const _Ty &
#line 199
get(const array< _Ty, _Size>  & _Arr) noexcept; 
#line 202
template< size_t _Idx, class _Ty, size_t _Size> 
#line 203
[[nodiscard]] constexpr _Ty &&
#line 202
get(array< _Ty, _Size>  && _Arr) noexcept; 
#line 205
template< size_t _Idx, class _Ty, size_t _Size> 
#line 206
[[nodiscard]] constexpr const _Ty &&
#line 205
get(const array< _Ty, _Size>  && _Arr) noexcept; 
#line 246 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
template < class _Ty1, class _Ty2 >
struct pair {
    using first_type = _Ty1;
    using second_type = _Ty2;

    template < class _Uty1 = _Ty1, class _Uty2 = _Ty2,
        enable_if_t < conjunction_v < is_default_constructible < _Uty1 >, is_default_constructible < _Uty2 >>, int > = 0 >
    constexpr explicit (
        ! conjunction_v < _Is_implicitly_default_constructible < _Uty1 >, _Is_implicitly_default_constructible < _Uty2 >> ) pair ( )
        noexcept ( is_nothrow_default_constructible_v < _Uty1 > && is_nothrow_default_constructible_v < _Uty2 > )
        : first ( ), second ( ) { }

    template < class _Uty1 = _Ty1, class _Uty2 = _Ty2,
        enable_if_t < conjunction_v < is_copy_constructible < _Uty1 >, is_copy_constructible < _Uty2 >>, int > = 0 >
    constexpr explicit ( ! conjunction_v < is_convertible < const _Uty1 &, _Uty1 >, is_convertible < const _Uty2 &, _Uty2 >> )
        pair ( const _Ty1 & _Val1, const _Ty2 & _Val2 )
            noexcept ( is_nothrow_copy_constructible_v < _Uty1 > && is_nothrow_copy_constructible_v < _Uty2 > )
        : first ( _Val1 ), second ( _Val2 ) { }




    template < class _Other1, class _Other2,

        enable_if_t < conjunction_v < is_constructible < _Ty1, _Other1 >, is_constructible < _Ty2, _Other2 >>, int > = 0 >
    constexpr explicit ( ! conjunction_v < is_convertible < _Other1, _Ty1 >, is_convertible < _Other2, _Ty2 >> )
        pair ( _Other1 && _Val1, _Other2 && _Val2 ) noexcept (
            is_nothrow_constructible_v < _Ty1, _Other1 > && is_nothrow_constructible_v < _Ty2, _Other2 > )
        : first ( :: std :: forward < _Other1 > ( _Val1 ) ), second ( :: std :: forward < _Other2 > ( _Val2 ) ) {
    }

    pair ( const pair & ) = default;
    pair ( pair && ) = default;










    template < class _Other1, class _Other2,
        enable_if_t < conjunction_v < is_constructible < _Ty1, const _Other1 & >, is_constructible < _Ty2, const _Other2 & >>,
            int > = 0 >
    constexpr explicit ( ! conjunction_v < is_convertible < const _Other1 &, _Ty1 >, is_convertible < const _Other2 &, _Ty2 >> )
        pair ( const pair < _Other1, _Other2 > & _Right )
            noexcept ( is_nothrow_constructible_v < _Ty1, const _Other1 & >
                     && is_nothrow_constructible_v < _Ty2, const _Other2 & > )
        : first ( _Right . first ), second ( _Right . second ) { }

    template < class _Other1, class _Other2,
        enable_if_t < conjunction_v < is_constructible < _Ty1, _Other1 >, is_constructible < _Ty2, _Other2 >>, int > = 0 >
    constexpr explicit ( ! conjunction_v < is_convertible < _Other1, _Ty1 >, is_convertible < _Other2, _Ty2 >> )
        pair ( pair < _Other1, _Other2 > && _Right ) noexcept (
            is_nothrow_constructible_v < _Ty1, _Other1 > && is_nothrow_constructible_v < _Ty2, _Other2 > )
        : first ( :: std :: forward < _Other1 > ( _Right . first ) ), second ( :: std :: forward < _Other2 > ( _Right . second ) ) { }

























    template < class ... _Types1, class ... _Types2 >
    inline pair ( piecewise_construct_t, tuple < _Types1 ... > _Val1, tuple < _Types2 ... > _Val2 )
        : pair ( _Val1, _Val2, index_sequence_for < _Types1 ... > { }, index_sequence_for < _Types2 ... > { } ) { }

    pair & operator = ( const volatile pair & ) = delete;

    template < class _Myself = pair,
        enable_if_t < conjunction_v < _Is_copy_assignable_no_precondition_check < typename _Myself :: first_type >,
                        _Is_copy_assignable_no_precondition_check < typename _Myself :: second_type >>,
            int > = 0 >
    inline pair & operator = ( _Identity_t < const _Myself & > _Right )
        noexcept ( conjunction_v < is_nothrow_copy_assignable < _Ty1 >, is_nothrow_copy_assignable < _Ty2 >> ) {
        first = _Right . first;
        second = _Right . second;
        return * this;
    }














    template < class _Myself = pair,
        enable_if_t < conjunction_v < _Is_move_assignable_no_precondition_check < typename _Myself :: first_type >,
                        _Is_move_assignable_no_precondition_check < typename _Myself :: second_type >>,
            int > = 0 >
    inline pair & operator = ( _Identity_t < _Myself && > _Right )
        noexcept ( conjunction_v < is_nothrow_move_assignable < _Ty1 >, is_nothrow_move_assignable < _Ty2 >> ) {
        first = :: std :: forward < _Ty1 > ( _Right . first );
        second = :: std :: forward < _Ty2 > ( _Right . second );
        return * this;
    }














    template < class _Other1, class _Other2,
        enable_if_t < conjunction_v < negation < is_same < pair, pair < _Other1, _Other2 >> >, is_assignable < _Ty1 &, const _Other1 & >,
                        is_assignable < _Ty2 &, const _Other2 & >>,
            int > = 0 >
    inline pair & operator = ( const pair < _Other1, _Other2 > & _Right )
        noexcept ( is_nothrow_assignable_v < _Ty1 &, const _Other1 & >
                 && is_nothrow_assignable_v < _Ty2 &, const _Other2 & > ) {
        first = _Right . first;
        second = _Right . second;
        return * this;
    }














    template < class _Other1, class _Other2,
        enable_if_t < conjunction_v < negation < is_same < pair, pair < _Other1, _Other2 >> >, is_assignable < _Ty1 &, _Other1 >,
                        is_assignable < _Ty2 &, _Other2 >>,
            int > = 0 >
    inline pair & operator = ( pair < _Other1, _Other2 > && _Right ) noexcept (
        is_nothrow_assignable_v < _Ty1 &, _Other1 > && is_nothrow_assignable_v < _Ty2 &, _Other2 > ) {
        first = :: std :: forward < _Other1 > ( _Right . first );
        second = :: std :: forward < _Other2 > ( _Right . second );
        return * this;
    }





































    inline void swap ( pair & _Right )
        noexcept ( _Is_nothrow_swappable < _Ty1 > :: value && _Is_nothrow_swappable < _Ty2 > :: value ) {
        using :: std :: swap;
        swap ( first, _Right . first );
        swap ( second, _Right . second );
    }











    _Ty1 first;
    _Ty2 second;

private :
    template < class _Tuple1, class _Tuple2, size_t ... _Indices1, size_t ... _Indices2 >
    constexpr pair ( _Tuple1 & _Val1, _Tuple2 & _Val2, index_sequence < _Indices1 ... >, index_sequence < _Indices2 ... > )
        : first ( :: std :: _Tuple_get < _Indices1 > ( :: std :: move ( _Val1 ) ) ... ),
          second ( :: std :: _Tuple_get < _Indices2 > ( :: std :: move ( _Val2 ) ) ... ) { }
};
#line 487 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
template < class _Ty1, class _Ty2,
    enable_if_t < _Is_swappable < _Ty1 > :: value && _Is_swappable < _Ty2 > :: value, int > = 0 >
inline void swap ( pair < _Ty1, _Ty2 > & _Left, pair < _Ty1, _Ty2 > & _Right ) noexcept ( noexcept ( _Left . swap ( _Right ) ) ) {
    _Left . swap ( _Right );
}
#line 502 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
template < class _Ty1, class _Ty2, class _Uty1, class _Uty2 >
[ [ nodiscard ] ] constexpr bool operator == ( const pair < _Ty1, _Ty2 > & _Left, const pair < _Uty1, _Uty2 > & _Right ) {
    return _Left . first == _Right . first && _Left . second == _Right . second;
}
#line 518 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
template < class _Ty1, class _Ty2, class _Uty1, class _Uty2 >
[ [ nodiscard ] ] constexpr bool operator != ( const pair < _Ty1, _Ty2 > & _Left, const pair < _Uty1, _Uty2 > & _Right ) {
    return ! ( _Left == _Right );
}
#line 523
template < class _Ty1, class _Ty2, class _Uty1, class _Uty2 >
[ [ nodiscard ] ] constexpr bool operator < ( const pair < _Ty1, _Ty2 > & _Left, const pair < _Uty1, _Uty2 > & _Right ) {
    return _Left . first < _Right . first || ( ! ( _Right . first < _Left . first ) && _Left . second < _Right . second );
}
#line 528
template < class _Ty1, class _Ty2, class _Uty1, class _Uty2 >
[ [ nodiscard ] ] constexpr bool operator > ( const pair < _Ty1, _Ty2 > & _Left, const pair < _Uty1, _Uty2 > & _Right ) {
    return _Right < _Left;
}
#line 533
template < class _Ty1, class _Ty2, class _Uty1, class _Uty2 >
[ [ nodiscard ] ] constexpr bool operator <= ( const pair < _Ty1, _Ty2 > & _Left, const pair < _Uty1, _Uty2 > & _Right ) {
    return ! ( _Right < _Left );
}
#line 538
template < class _Ty1, class _Ty2, class _Uty1, class _Uty2 >
[ [ nodiscard ] ] constexpr bool operator >= ( const pair < _Ty1, _Ty2 > & _Left, const pair < _Uty1, _Uty2 > & _Right ) {
    return ! ( _Left < _Right );
}
#line 561 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
template< class _Ty> 
#line 562
struct _Unrefwrap_helper { 
#line 563
using type = _Ty; 
#line 564
}; 
#line 566
template< class _Ty> 
#line 567
struct _Unrefwrap_helper< reference_wrapper< _Ty> >  { 
#line 568
using type = _Ty &; 
#line 569
}; 
#line 572
template< class _Ty> using _Unrefwrap_t = typename _Unrefwrap_helper< decay_t< _Ty> > ::type; 
#line 575
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr pair < _Unrefwrap_t < _Ty1 >, _Unrefwrap_t < _Ty2 > > make_pair ( _Ty1 && _Val1, _Ty2 && _Val2 )
    noexcept ( is_nothrow_constructible_v < _Unrefwrap_t < _Ty1 >, _Ty1 >
             && is_nothrow_constructible_v < _Unrefwrap_t < _Ty2 >, _Ty2 > ) {

    using _Mypair = pair < _Unrefwrap_t < _Ty1 >, _Unrefwrap_t < _Ty2 >>;
    return _Mypair ( :: std :: forward < _Ty1 > ( _Val1 ), :: std :: forward < _Ty2 > ( _Val2 ) );
}
#line 584
namespace rel_ops { 
#line 585
template < class _Ty >
     [ [ nodiscard ] ] bool operator != ( const _Ty & _Left, const _Ty & _Right ) {
        return ! ( _Left == _Right );
    }
#line 590
template < class _Ty >
     [ [ nodiscard ] ] bool operator > ( const _Ty & _Left, const _Ty & _Right ) {
        return _Right < _Left;
    }
#line 595
template < class _Ty >
     [ [ nodiscard ] ] bool operator <= ( const _Ty & _Left, const _Ty & _Right ) {
        return ! ( _Right < _Left );
    }
#line 600
template < class _Ty >
     [ [ nodiscard ] ] bool operator >= ( const _Ty & _Left, const _Ty & _Right ) {
        return ! ( _Left < _Right );
    }
#line 604
}
#line 606
template< class _Tuple, class  = void> 
#line 607
struct _Tuple_size_sfinae { }; 
#line 609
template< class _Tuple> 
#line 610
struct _Tuple_size_sfinae< _Tuple, void_t< decltype(tuple_size< _Tuple> ::value)> >  : public integral_constant< unsigned __int64, tuple_size< _Tuple> ::value>  { 
#line 611
}; 
#line 613
template< class _Tuple> 
#line 614
struct tuple_size< const _Tuple>  : public _Tuple_size_sfinae< _Tuple>  { }; 
#line 616
template< class _Tuple> 
#line 617
struct tuple_size< volatile _Tuple>  : public _Tuple_size_sfinae< _Tuple>  { }; 
#line 619
template< class _Tuple> 
#line 620
struct tuple_size< const volatile _Tuple>  : public _Tuple_size_sfinae< _Tuple>  { }; 
#line 622
template< size_t _Index, class _Tuple> 
#line 623
struct tuple_element< _Index, const _Tuple>  : public std::tuple_element< _Index, _Tuple>  { 
#line 624
using _Mybase = std::tuple_element< _Index, _Tuple> ; 
#line 625
using type = std::add_const_t< typename std::tuple_element< _Index, _Tuple> ::type> ; 
#line 626
}; 
#line 628
template< size_t _Index, class _Tuple> 
#line 629
struct tuple_element< _Index, volatile _Tuple>  : public std::tuple_element< _Index, _Tuple>  { 
#line 631
using _Mybase = std::tuple_element< _Index, _Tuple> ; 
#line 632
using type = std::add_volatile_t< typename std::tuple_element< _Index, _Tuple> ::type> ; 
#line 633
}; 
#line 635
template< size_t _Index, class _Tuple> 
#line 636
struct tuple_element< _Index, const volatile _Tuple>  : public std::tuple_element< _Index, _Tuple>  { 
#line 638
using _Mybase = std::tuple_element< _Index, _Tuple> ; 
#line 639
using type = std::add_cv_t< typename std::tuple_element< _Index, _Tuple> ::type> ; 
#line 640
}; 
#line 642
template< class _Ty, size_t _Size> 
#line 643
struct tuple_size< array< _Ty, _Size> >  : public integral_constant< unsigned __int64, _Size>  { }; 
#line 645
template< size_t _Idx, class _Ty, size_t _Size> 
#line 646
struct tuple_element< _Idx, array< _Ty, _Size> >  { 
#line 647
static_assert((_Idx < _Size), "array index out of bounds");
#line 649
using type = _Ty; 
#line 650
}; 
#line 652
template< class ..._Types> 
#line 653
struct tuple_size< tuple< _Types...> >  : public integral_constant< unsigned __int64, sizeof...(_Types)>  { }; 
#line 655
template< size_t _Index> 
#line 656
struct tuple_element< _Index, tuple< > >  { 
#line 657
static_assert((_Always_false< integral_constant< size_t, _Index> > ), "tuple index out of bounds");
#line 658
}; 
#line 660
template< class _This, class ..._Rest> 
#line 661
struct tuple_element< 0, tuple< _This, _Rest...> >  { 
#line 662
using type = _This; 
#line 664
using _Ttype = tuple< _This, _Rest...> ; 
#line 665
}; 
#line 667
template< size_t _Index, class _This, class ..._Rest> 
#line 668
struct tuple_element< _Index, tuple< _This, _Rest...> >  : public std::tuple_element< _Index - (1), tuple< _Rest...> >  { 
#line 669
}; 
#line 671
template< class _Ty1, class _Ty2> 
#line 672
struct tuple_size< pair< _Ty1, _Ty2> >  : public integral_constant< unsigned __int64, 2Ui64>  { }; 
#line 674
template< size_t _Idx, class _Ty1, class _Ty2> 
#line 675
struct tuple_element< _Idx, pair< _Ty1, _Ty2> >  { 
#line 676
static_assert((_Idx < (2)), "pair index out of bounds");
#line 678
using type = conditional_t< _Idx == (0), _Ty1, _Ty2> ; 
#line 679
}; 
#line 681
template < size_t _Idx, class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr tuple_element_t < _Idx, pair < _Ty1, _Ty2 > > & get ( pair < _Ty1, _Ty2 > & _Pr ) noexcept {

    if constexpr ( _Idx == 0 ) {
        return _Pr . first;
    } else {
        return _Pr . second;
    }
}
#line 691
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr _Ty1 & get ( pair < _Ty1, _Ty2 > & _Pr ) noexcept {

    return _Pr . first;
}
#line 697
template < class _Ty2, class _Ty1 >
[ [ nodiscard ] ] constexpr _Ty2 & get ( pair < _Ty1, _Ty2 > & _Pr ) noexcept {

    return _Pr . second;
}
#line 703
template < size_t _Idx, class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr const tuple_element_t < _Idx, pair < _Ty1, _Ty2 > > & get ( const pair < _Ty1, _Ty2 > & _Pr ) noexcept {

    if constexpr ( _Idx == 0 ) {
        return _Pr . first;
    } else {
        return _Pr . second;
    }
}
#line 713
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr const _Ty1 & get ( const pair < _Ty1, _Ty2 > & _Pr ) noexcept {

    return _Pr . first;
}
#line 719
template < class _Ty2, class _Ty1 >
[ [ nodiscard ] ] constexpr const _Ty2 & get ( const pair < _Ty1, _Ty2 > & _Pr ) noexcept {

    return _Pr . second;
}
#line 725
template < size_t _Idx, class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr tuple_element_t < _Idx, pair < _Ty1, _Ty2 > > && get ( pair < _Ty1, _Ty2 > && _Pr ) noexcept {

    if constexpr ( _Idx == 0 ) {
        return :: std :: forward < _Ty1 > ( _Pr . first );
    } else {
        return :: std :: forward < _Ty2 > ( _Pr . second );
    }
}
#line 735
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr _Ty1 && get ( pair < _Ty1, _Ty2 > && _Pr ) noexcept {

    return :: std :: forward < _Ty1 > ( _Pr . first );
}
#line 741
template < class _Ty2, class _Ty1 >
[ [ nodiscard ] ] constexpr _Ty2 && get ( pair < _Ty1, _Ty2 > && _Pr ) noexcept {

    return :: std :: forward < _Ty2 > ( _Pr . second );
}
#line 747
template < size_t _Idx, class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr const tuple_element_t < _Idx, pair < _Ty1, _Ty2 > > && get ( const pair < _Ty1, _Ty2 > && _Pr ) noexcept {

    if constexpr ( _Idx == 0 ) {
        return :: std :: forward < const _Ty1 > ( _Pr . first );
    } else {
        return :: std :: forward < const _Ty2 > ( _Pr . second );
    }
}
#line 757
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr const _Ty1 && get ( const pair < _Ty1, _Ty2 > && _Pr ) noexcept {

    return :: std :: forward < const _Ty1 > ( _Pr . first );
}
#line 763
template < class _Ty2, class _Ty1 >
[ [ nodiscard ] ] constexpr const _Ty2 && get ( const pair < _Ty1, _Ty2 > && _Pr ) noexcept {

    return :: std :: forward < const _Ty2 > ( _Pr . second );
}
#line 769
template < class _Ty, class _Other = _Ty >
inline _Ty exchange ( _Ty & _Val, _Other && _New_val )
    noexcept ( conjunction_v < is_nothrow_move_constructible < _Ty >, is_nothrow_assignable < _Ty &, _Other >> ) {

    _Ty _Old_val = static_cast < _Ty && > ( _Val );
    _Val = static_cast < _Other && > ( _New_val );
    return _Old_val;
}
#line 778
template < class _Ty >
[ [ nodiscard ] ] constexpr add_const_t < _Ty > & as_const ( _Ty & _Val ) noexcept {
    return _Val;
}
#line 783
template < class _Ty >
void as_const ( const _Ty && ) = delete;
#line 807 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr bool _Cmp_equal ( const _Ty1 _Left, const _Ty2 _Right ) noexcept {
    ;
    if constexpr ( is_signed_v < _Ty1 > == is_signed_v < _Ty2 > ) {
        return _Left == _Right;
    } else if constexpr ( is_signed_v < _Ty2 > ) {
        return _Left == static_cast < make_unsigned_t < _Ty2 >> ( _Right ) && _Right >= 0;
    } else {
        return static_cast < make_unsigned_t < _Ty1 >> ( _Left ) == _Right && _Left >= 0;
    }
}
#line 819
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr bool _Cmp_not_equal ( const _Ty1 _Left, const _Ty2 _Right ) noexcept {
    return ! :: std :: _Cmp_equal ( _Left, _Right );
}
#line 824
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr bool _Cmp_less ( const _Ty1 _Left, const _Ty2 _Right ) noexcept {
    ;
    if constexpr ( is_signed_v < _Ty1 > == is_signed_v < _Ty2 > ) {
        return _Left < _Right;
    } else if constexpr ( is_signed_v < _Ty2 > ) {
        return _Right > 0 && _Left < static_cast < make_unsigned_t < _Ty2 >> ( _Right );
    } else {
        return _Left < 0 || static_cast < make_unsigned_t < _Ty1 >> ( _Left ) < _Right;
    }
}
#line 836
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr bool _Cmp_greater ( const _Ty1 _Left, const _Ty2 _Right ) noexcept {
    return :: std :: _Cmp_less ( _Right, _Left );
}
#line 841
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr bool _Cmp_less_equal ( const _Ty1 _Left, const _Ty2 _Right ) noexcept {
    return ! :: std :: _Cmp_less ( _Right, _Left );
}
#line 846
template < class _Ty1, class _Ty2 >
[ [ nodiscard ] ] constexpr bool _Cmp_greater_equal ( const _Ty1 _Left, const _Ty2 _Right ) noexcept {
    return ! :: std :: _Cmp_less ( _Left, _Right );
}
#line 851
template < class _Ty >
[ [ nodiscard ] ] constexpr _Ty _Min_limit ( ) noexcept {
    ;
    if constexpr ( is_signed_v < _Ty > ) {
        constexpr auto _Unsigned_max = static_cast < make_unsigned_t < _Ty >> ( - 1 );
        return static_cast < _Ty > ( ( _Unsigned_max >> 1 ) + 1 );
    } else {
        return 0;
    }
}
#line 862
template < class _Ty >
[ [ nodiscard ] ] constexpr _Ty _Max_limit ( ) noexcept {
    ;
    if constexpr ( is_signed_v < _Ty > ) {
        constexpr auto _Unsigned_max = static_cast < make_unsigned_t < _Ty >> ( - 1 );
        return static_cast < _Ty > ( _Unsigned_max >> 1 );
    } else {
        return static_cast < _Ty > ( - 1 );
    }
}
#line 873
template < class _Rx, class _Ty >
[ [ nodiscard ] ] constexpr bool _In_range ( const _Ty _Value ) noexcept {
    ;

    constexpr auto _Ty_min = _Min_limit < _Ty > ( );
    constexpr auto _Rx_min = _Min_limit < _Rx > ( );

    if constexpr ( :: std :: _Cmp_less ( _Ty_min, _Rx_min ) ) {
        if ( _Value < _Ty { _Rx_min } ) {
            return false;
        }
    }

    constexpr auto _Ty_max = _Max_limit < _Ty > ( );
    constexpr auto _Rx_max = _Max_limit < _Rx > ( );

    if constexpr ( :: std :: _Cmp_greater ( _Ty_max, _Rx_max ) ) {
        if ( _Value > _Ty { _Rx_max } ) {
            return false;
        }
    }

    return true;
}
#line 977 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
namespace [[deprecated("warning STL4002: The non-Standard std::tr1 namespace and TR1-only machinery are deprecated and will be REMOVED. You can define _" "SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING to suppress this warning.")]] tr1 { 
#line 978
using std::get;
#line 979
using std::tuple_element;
#line 980
using std::tuple_size;
#line 981
}
#line 984 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.43.34808\\include\\utility"
}
#line 994
#pragma warning(pop)
#pragma pack ( pop )
#line 207 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 208
cudaLaunchKernel(T *
#line 209
func, ::dim3 
#line 210
gridDim, ::dim3 
#line 211
blockDim, void **
#line 212
args, ::size_t 
#line 213
sharedMem = 0, ::cudaStream_t 
#line 214
stream = 0) 
#line 216
{ 
#line 217
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 218
} 
#line 279
template< class ...ExpTypes, class ...ActTypes> static __inline ::cudaError_t 
#line 280
cudaLaunchKernelEx(const ::cudaLaunchConfig_t *
#line 281
config, void (*
#line 282
kernel)(ExpTypes ...), ActTypes &&...
#line 283
args) 
#line 285
{ 
#line 286
return [&](ExpTypes ...coercedArgs) { 
#line 287
void *pArgs[] = {(&coercedArgs)...}; 
#line 288
return ::cudaLaunchKernelExC(config, (const void *)(kernel), pArgs); 
#line 289
} (std::forward< ActTypes> (args)...); 
#line 290
} 
#line 348
template< class ...ActTypes> static __inline ::cudaError_t 
#line 349
cudaLaunchKernelEx(const ::cudaLaunchConfig_t *
#line 350
config, const ::cudaKernel_t 
#line 351
kernel, ActTypes &&...
#line 352
args) 
#line 354
{ 
#line 355
void *pArgs[] = {(&args)...}; 
#line 356
return ::cudaLaunchKernelExC(config, (const void *)kernel, pArgs); 
#line 357
} 
#line 410 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 411
cudaLaunchCooperativeKernel(T *
#line 412
func, ::dim3 
#line 413
gridDim, ::dim3 
#line 414
blockDim, void **
#line 415
args, ::size_t 
#line 416
sharedMem = 0, ::cudaStream_t 
#line 417
stream = 0) 
#line 419
{ 
#line 420
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 421
} 
#line 454
static __inline cudaError_t cudaEventCreate(cudaEvent_t *
#line 455
event, unsigned 
#line 456
flags) 
#line 458
{ 
#line 459
return ::cudaEventCreateWithFlags(event, flags); 
#line 460
} 
#line 498
static __inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *
#line 499
pGraphExec, cudaGraph_t 
#line 500
graph, cudaGraphNode_t *
#line 501
pErrorNode, char *
#line 502
pLogBuffer, size_t 
#line 503
bufferSize) 
#line 505
{ 
#line 506
(void)pErrorNode; 
#line 507
(void)pLogBuffer; 
#line 508
(void)bufferSize; 
#line 509
return ::cudaGraphInstantiate(pGraphExec, graph, 0); 
#line 510
} 
#line 569
static __inline cudaError_t cudaMallocHost(void **
#line 570
ptr, size_t 
#line 571
size, unsigned 
#line 572
flags) 
#line 574
{ 
#line 575
return ::cudaHostAlloc(ptr, size, flags); 
#line 576
} 
#line 578
template< class T> static __inline ::cudaError_t 
#line 579
cudaHostAlloc(T **
#line 580
ptr, ::size_t 
#line 581
size, unsigned 
#line 582
flags) 
#line 584
{ 
#line 585
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
#line 586
} 
#line 588
template< class T> static __inline ::cudaError_t 
#line 589
cudaHostGetDevicePointer(T **
#line 590
pDevice, void *
#line 591
pHost, unsigned 
#line 592
flags) 
#line 594
{ 
#line 595
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
#line 596
} 
#line 698
template< class T> static __inline ::cudaError_t 
#line 699
cudaMallocManaged(T **
#line 700
devPtr, ::size_t 
#line 701
size, unsigned 
#line 702
flags = 1) 
#line 704
{ 
#line 705
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
#line 706
} 
#line 716
template< class T> ::cudaError_t 
#line 717
cudaMemAdvise(T *
#line 718
devPtr, ::size_t 
#line 719
count, ::cudaMemoryAdvise 
#line 720
advice, ::cudaMemLocation 
#line 721
location) 
#line 723
{ 
#line 724
return ::cudaMemAdvise_v2((const void *)devPtr, count, advice, location); 
#line 725
} 
#line 727
template< class T> static __inline ::cudaError_t 
#line 728
cudaMemPrefetchAsync(T *
#line 729
devPtr, ::size_t 
#line 730
count, ::cudaMemLocation 
#line 731
location, unsigned 
#line 732
flags, ::cudaStream_t 
#line 733
stream = 0) 
#line 735
{ 
#line 736
return ::cudaMemPrefetchAsync_v2((const void *)devPtr, count, location, flags, stream); 
#line 737
} 
#line 819
template< class T> static __inline ::cudaError_t 
#line 820
cudaStreamAttachMemAsync(::cudaStream_t 
#line 821
stream, T *
#line 822
devPtr, ::size_t 
#line 823
length = 0, unsigned 
#line 824
flags = 4) 
#line 826
{ 
#line 827
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
#line 828
} 
#line 830
template< class T> __inline ::cudaError_t 
#line 831
cudaMalloc(T **
#line 832
devPtr, ::size_t 
#line 833
size) 
#line 835
{ 
#line 836
return ::cudaMalloc((void **)((void *)devPtr), size); 
#line 837
} 
#line 839
template< class T> static __inline ::cudaError_t 
#line 840
cudaMallocHost(T **
#line 841
ptr, ::size_t 
#line 842
size, unsigned 
#line 843
flags = 0) 
#line 845
{ 
#line 846
return cudaMallocHost((void **)((void *)ptr), size, flags); 
#line 847
} 
#line 849
template< class T> static __inline ::cudaError_t 
#line 850
cudaMallocPitch(T **
#line 851
devPtr, ::size_t *
#line 852
pitch, ::size_t 
#line 853
width, ::size_t 
#line 854
height) 
#line 856
{ 
#line 857
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
#line 858
} 
#line 869
static __inline cudaError_t cudaMallocAsync(void **
#line 870
ptr, size_t 
#line 871
size, cudaMemPool_t 
#line 872
memPool, cudaStream_t 
#line 873
stream) 
#line 875
{ 
#line 876
return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream); 
#line 877
} 
#line 879
template< class T> static __inline ::cudaError_t 
#line 880
cudaMallocAsync(T **
#line 881
ptr, ::size_t 
#line 882
size, ::cudaMemPool_t 
#line 883
memPool, ::cudaStream_t 
#line 884
stream) 
#line 886
{ 
#line 887
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
#line 888
} 
#line 890
template< class T> static __inline ::cudaError_t 
#line 891
cudaMallocAsync(T **
#line 892
ptr, ::size_t 
#line 893
size, ::cudaStream_t 
#line 894
stream) 
#line 896
{ 
#line 897
return ::cudaMallocAsync((void **)((void *)ptr), size, stream); 
#line 898
} 
#line 900
template< class T> static __inline ::cudaError_t 
#line 901
cudaMallocFromPoolAsync(T **
#line 902
ptr, ::size_t 
#line 903
size, ::cudaMemPool_t 
#line 904
memPool, ::cudaStream_t 
#line 905
stream) 
#line 907
{ 
#line 908
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
#line 909
} 
#line 919
template< class T, class U> static __inline ::cudaError_t 
#line 920
cudaMemcpyBatchAsync(T **
#line 921
dsts, U **srcs, ::size_t *sizes, ::size_t count, ::cudaMemcpyAttributes *attrs, ::size_t *
#line 922
attrsIdxs, ::size_t numAttrs, ::size_t *failIdx, ::cudaStream_t hStream) 
#line 924
{ 
#line 925
return ::cudaMemcpyBatchAsync((void **)dsts, (void **)srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, hStream); 
#line 926
} 
#line 938
template< class T, class U> static __inline ::cudaError_t 
#line 939
cudaMemcpyBatchAsync(T **
#line 940
dsts, U **srcs, ::size_t *sizes, ::size_t count, ::cudaMemcpyAttributes attr, ::size_t *failIdx, ::cudaStream_t hStream) 
#line 942
{ 
#line 943
::size_t attrsIdxs = (0); 
#line 944
return ::cudaMemcpyBatchAsync((void **)dsts, (void **)srcs, sizes, count, &attr, &attrsIdxs, 1, failIdx, hStream); 
#line 945
} 
#line 984
template< class T> static __inline ::cudaError_t 
#line 985
cudaMemcpyToSymbol(const T &
#line 986
symbol, const void *
#line 987
src, ::size_t 
#line 988
count, ::size_t 
#line 989
offset = 0, ::cudaMemcpyKind 
#line 990
kind = cudaMemcpyHostToDevice) 
#line 992
{ 
#line 993
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
#line 994
} 
#line 1038
template< class T> static __inline ::cudaError_t 
#line 1039
cudaMemcpyToSymbolAsync(const T &
#line 1040
symbol, const void *
#line 1041
src, ::size_t 
#line 1042
count, ::size_t 
#line 1043
offset = 0, ::cudaMemcpyKind 
#line 1044
kind = cudaMemcpyHostToDevice, ::cudaStream_t 
#line 1045
stream = 0) 
#line 1047
{ 
#line 1048
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
#line 1049
} 
#line 1086
template< class T> static __inline ::cudaError_t 
#line 1087
cudaMemcpyFromSymbol(void *
#line 1088
dst, const T &
#line 1089
symbol, ::size_t 
#line 1090
count, ::size_t 
#line 1091
offset = 0, ::cudaMemcpyKind 
#line 1092
kind = cudaMemcpyDeviceToHost) 
#line 1094
{ 
#line 1095
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
#line 1096
} 
#line 1140
template< class T> static __inline ::cudaError_t 
#line 1141
cudaMemcpyFromSymbolAsync(void *
#line 1142
dst, const T &
#line 1143
symbol, ::size_t 
#line 1144
count, ::size_t 
#line 1145
offset = 0, ::cudaMemcpyKind 
#line 1146
kind = cudaMemcpyDeviceToHost, ::cudaStream_t 
#line 1147
stream = 0) 
#line 1149
{ 
#line 1150
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
#line 1151
} 
#line 1209
template< class T> static __inline ::cudaError_t 
#line 1210
cudaGraphAddMemcpyNodeToSymbol(::cudaGraphNode_t *
#line 1211
pGraphNode, ::cudaGraph_t 
#line 1212
graph, const ::cudaGraphNode_t *
#line 1213
pDependencies, ::size_t 
#line 1214
numDependencies, const T &
#line 1215
symbol, const void *
#line 1216
src, ::size_t 
#line 1217
count, ::size_t 
#line 1218
offset, ::cudaMemcpyKind 
#line 1219
kind) 
#line 1220
{ 
#line 1221
return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void *)(&symbol), src, count, offset, kind); 
#line 1222
} 
#line 1280
template< class T> static __inline ::cudaError_t 
#line 1281
cudaGraphAddMemcpyNodeFromSymbol(::cudaGraphNode_t *
#line 1282
pGraphNode, ::cudaGraph_t 
#line 1283
graph, const ::cudaGraphNode_t *
#line 1284
pDependencies, ::size_t 
#line 1285
numDependencies, void *
#line 1286
dst, const T &
#line 1287
symbol, ::size_t 
#line 1288
count, ::size_t 
#line 1289
offset, ::cudaMemcpyKind 
#line 1290
kind) 
#line 1291
{ 
#line 1292
return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void *)(&symbol), count, offset, kind); 
#line 1293
} 
#line 1331
template< class T> static __inline ::cudaError_t 
#line 1332
cudaGraphMemcpyNodeSetParamsToSymbol(::cudaGraphNode_t 
#line 1333
node, const T &
#line 1334
symbol, const void *
#line 1335
src, ::size_t 
#line 1336
count, ::size_t 
#line 1337
offset, ::cudaMemcpyKind 
#line 1338
kind) 
#line 1339
{ 
#line 1340
return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void *)(&symbol), src, count, offset, kind); 
#line 1341
} 
#line 1379
template< class T> static __inline ::cudaError_t 
#line 1380
cudaGraphMemcpyNodeSetParamsFromSymbol(::cudaGraphNode_t 
#line 1381
node, void *
#line 1382
dst, const T &
#line 1383
symbol, ::size_t 
#line 1384
count, ::size_t 
#line 1385
offset, ::cudaMemcpyKind 
#line 1386
kind) 
#line 1387
{ 
#line 1388
return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void *)(&symbol), count, offset, kind); 
#line 1389
} 
#line 1437
template< class T> static __inline ::cudaError_t 
#line 1438
cudaGraphExecMemcpyNodeSetParamsToSymbol(::cudaGraphExec_t 
#line 1439
hGraphExec, ::cudaGraphNode_t 
#line 1440
node, const T &
#line 1441
symbol, const void *
#line 1442
src, ::size_t 
#line 1443
count, ::size_t 
#line 1444
offset, ::cudaMemcpyKind 
#line 1445
kind) 
#line 1446
{ 
#line 1447
return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void *)(&symbol), src, count, offset, kind); 
#line 1448
} 
#line 1496
template< class T> static __inline ::cudaError_t 
#line 1497
cudaGraphExecMemcpyNodeSetParamsFromSymbol(::cudaGraphExec_t 
#line 1498
hGraphExec, ::cudaGraphNode_t 
#line 1499
node, void *
#line 1500
dst, const T &
#line 1501
symbol, ::size_t 
#line 1502
count, ::size_t 
#line 1503
offset, ::cudaMemcpyKind 
#line 1504
kind) 
#line 1505
{ 
#line 1506
return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void *)(&symbol), count, offset, kind); 
#line 1507
} 
#line 1510
static __inline cudaError_t __stdcall cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out) 
#line 1511
{ 
#line 1512
cudaGraphExecUpdateResultInfo resultInfo; 
#line 1513
cudaError_t status = cudaGraphExecUpdate(hGraphExec, hGraph, &resultInfo); 
#line 1514
if (hErrorNode_out) { 
#line 1515
(*hErrorNode_out) = (resultInfo.errorNode); 
#line 1516
}  
#line 1517
if (updateResult_out) { 
#line 1518
(*updateResult_out) = (resultInfo.result); 
#line 1519
}  
#line 1520
return status; 
#line 1521
} 
#line 1549
template< class T> static __inline ::cudaError_t 
#line 1550
cudaUserObjectCreate(::cudaUserObject_t *
#line 1551
object_out, T *
#line 1552
objectToWrap, unsigned 
#line 1553
initialRefcount, unsigned 
#line 1554
flags) 
#line 1555
{ 
#line 1556
return ::cudaUserObjectCreate(object_out, objectToWrap, [](void *
#line 1559
vpObj) { delete (reinterpret_cast< T *>(vpObj)); } , initialRefcount, flags); 
#line 1562
} 
#line 1564
template< class T> static __inline ::cudaError_t 
#line 1565
cudaUserObjectCreate(::cudaUserObject_t *
#line 1566
object_out, T *
#line 1567
objectToWrap, unsigned 
#line 1568
initialRefcount, ::cudaUserObjectFlags 
#line 1569
flags) 
#line 1570
{ 
#line 1571
return cudaUserObjectCreate(object_out, objectToWrap, initialRefcount, (unsigned)flags); 
#line 1572
} 
#line 1599 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 1600
cudaGetSymbolAddress(void **
#line 1601
devPtr, const T &
#line 1602
symbol) 
#line 1604
{ 
#line 1605
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
#line 1606
} 
#line 1631
template< class T> static __inline ::cudaError_t 
#line 1632
cudaGetSymbolSize(::size_t *
#line 1633
size, const T &
#line 1634
symbol) 
#line 1636
{ 
#line 1637
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
#line 1638
} 
#line 1683
template< class T> static __inline ::cudaError_t 
#line 1684
cudaFuncSetCacheConfig(T *
#line 1685
func, ::cudaFuncCache 
#line 1686
cacheConfig) 
#line 1688
{ 
#line 1689
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
#line 1690
} 
#line 1692
template< class T> 
#line 1694
__declspec(deprecated) static __inline ::cudaError_t 
#line 1695
cudaFuncSetSharedMemConfig(T *
#line 1696
func, ::cudaSharedMemConfig 
#line 1697
config) 
#line 1699
{ 
#line 1704 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
#pragma warning(suppress: 4996)
#line 1706 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
#line 1710 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
} 
#line 1743 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
template< class T> __inline ::cudaError_t 
#line 1744
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
#line 1745
numBlocks, T 
#line 1746
func, int 
#line 1747
blockSize, ::size_t 
#line 1748
dynamicSMemSize) 
#line 1749
{ 
#line 1750
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
#line 1751
} 
#line 1796
template< class T> __inline ::cudaError_t 
#line 1797
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
#line 1798
numBlocks, T 
#line 1799
func, int 
#line 1800
blockSize, ::size_t 
#line 1801
dynamicSMemSize, unsigned 
#line 1802
flags) 
#line 1803
{ 
#line 1804
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
#line 1805
} 
#line 1810
class __cudaOccupancyB2DHelper { 
#line 1811
size_t n; 
#line 1813
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
#line 1814
size_t operator()(int) 
#line 1815
{ 
#line 1816
return n; 
#line 1817
} 
#line 1818
}; 
#line 1866
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 1867
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
#line 1868
minGridSize, int *
#line 1869
blockSize, T 
#line 1870
func, UnaryFunction 
#line 1871
blockSizeToDynamicSMemSize, int 
#line 1872
blockSizeLimit = 0, unsigned 
#line 1873
flags = 0) 
#line 1874
{ 
#line 1875
::cudaError_t status; 
#line 1878
int device; 
#line 1879
::cudaFuncAttributes attr; 
#line 1882
int maxThreadsPerMultiProcessor; 
#line 1883
int warpSize; 
#line 1884
int devMaxThreadsPerBlock; 
#line 1885
int multiProcessorCount; 
#line 1886
int funcMaxThreadsPerBlock; 
#line 1887
int occupancyLimit; 
#line 1888
int granularity; 
#line 1891
int maxBlockSize = 0; 
#line 1892
int numBlocks = 0; 
#line 1893
int maxOccupancy = 0; 
#line 1896
int blockSizeToTryAligned; 
#line 1897
int blockSizeToTry; 
#line 1898
int blockSizeLimitAligned; 
#line 1899
int occupancyInBlocks; 
#line 1900
int occupancyInThreads; 
#line 1901
::size_t dynamicSMemSize; 
#line 1907
if (((!minGridSize) || (!blockSize)) || (!func)) { 
#line 1908
return cudaErrorInvalidValue; 
#line 1909
}  
#line 1915
status = ::cudaGetDevice(&device); 
#line 1916
if (status != (cudaSuccess)) { 
#line 1917
return status; 
#line 1918
}  
#line 1920
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
#line 1924
if (status != (cudaSuccess)) { 
#line 1925
return status; 
#line 1926
}  
#line 1928
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
#line 1932
if (status != (cudaSuccess)) { 
#line 1933
return status; 
#line 1934
}  
#line 1936
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
#line 1940
if (status != (cudaSuccess)) { 
#line 1941
return status; 
#line 1942
}  
#line 1944
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
#line 1948
if (status != (cudaSuccess)) { 
#line 1949
return status; 
#line 1950
}  
#line 1952
status = cudaFuncGetAttributes(&attr, func); 
#line 1953
if (status != (cudaSuccess)) { 
#line 1954
return status; 
#line 1955
}  
#line 1957
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
#line 1963
occupancyLimit = maxThreadsPerMultiProcessor; 
#line 1964
granularity = warpSize; 
#line 1966
if (blockSizeLimit == 0) { 
#line 1967
blockSizeLimit = devMaxThreadsPerBlock; 
#line 1968
}  
#line 1970
if (devMaxThreadsPerBlock < blockSizeLimit) { 
#line 1971
blockSizeLimit = devMaxThreadsPerBlock; 
#line 1972
}  
#line 1974
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
#line 1975
blockSizeLimit = funcMaxThreadsPerBlock; 
#line 1976
}  
#line 1978
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
#line 1980
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
#line 1984
if (blockSizeLimit < blockSizeToTryAligned) { 
#line 1985
blockSizeToTry = blockSizeLimit; 
#line 1986
} else { 
#line 1987
blockSizeToTry = blockSizeToTryAligned; 
#line 1988
}  
#line 1990
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
#line 1992
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
#line 1999
if (status != (cudaSuccess)) { 
#line 2000
return status; 
#line 2001
}  
#line 2003
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
#line 2005
if (occupancyInThreads > maxOccupancy) { 
#line 2006
maxBlockSize = blockSizeToTry; 
#line 2007
numBlocks = occupancyInBlocks; 
#line 2008
maxOccupancy = occupancyInThreads; 
#line 2009
}  
#line 2013
if (occupancyLimit == maxOccupancy) { 
#line 2014
break; 
#line 2015
}  
#line 2016
}  
#line 2024
(*minGridSize) = (numBlocks * multiProcessorCount); 
#line 2025
(*blockSize) = maxBlockSize; 
#line 2027
return status; 
#line 2028
} 
#line 2062
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 2063
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
#line 2064
minGridSize, int *
#line 2065
blockSize, T 
#line 2066
func, UnaryFunction 
#line 2067
blockSizeToDynamicSMemSize, int 
#line 2068
blockSizeLimit = 0) 
#line 2069
{ 
#line 2070
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
#line 2071
} 
#line 2108
template< class T> static __inline ::cudaError_t 
#line 2109
cudaOccupancyMaxPotentialBlockSize(int *
#line 2110
minGridSize, int *
#line 2111
blockSize, T 
#line 2112
func, ::size_t 
#line 2113
dynamicSMemSize = 0, int 
#line 2114
blockSizeLimit = 0) 
#line 2115
{ 
#line 2116
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
#line 2117
} 
#line 2147
template< class T> static __inline ::cudaError_t 
#line 2148
cudaOccupancyAvailableDynamicSMemPerBlock(::size_t *
#line 2149
dynamicSmemSize, T *
#line 2150
func, int 
#line 2151
numBlocks, int 
#line 2152
blockSize) 
#line 2153
{ 
#line 2154
return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void *)func, numBlocks, blockSize); 
#line 2155
} 
#line 2206
template< class T> static __inline ::cudaError_t 
#line 2207
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
#line 2208
minGridSize, int *
#line 2209
blockSize, T 
#line 2210
func, ::size_t 
#line 2211
dynamicSMemSize = 0, int 
#line 2212
blockSizeLimit = 0, unsigned 
#line 2213
flags = 0) 
#line 2214
{ 
#line 2215
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
#line 2216
} 
#line 2251
template< class T> static __inline ::cudaError_t 
#line 2252
cudaOccupancyMaxPotentialClusterSize(int *
#line 2253
clusterSize, T *
#line 2254
func, const ::cudaLaunchConfig_t *
#line 2255
config) 
#line 2256
{ 
#line 2257
return ::cudaOccupancyMaxPotentialClusterSize(clusterSize, (const void *)func, config); 
#line 2258
} 
#line 2295
template< class T> static __inline ::cudaError_t 
#line 2296
cudaOccupancyMaxActiveClusters(int *
#line 2297
numClusters, T *
#line 2298
func, const ::cudaLaunchConfig_t *
#line 2299
config) 
#line 2300
{ 
#line 2301
return ::cudaOccupancyMaxActiveClusters(numClusters, (const void *)func, config); 
#line 2302
} 
#line 2336
template< class T> __inline ::cudaError_t 
#line 2337
cudaFuncGetAttributes(::cudaFuncAttributes *
#line 2338
attr, T *
#line 2339
entry) 
#line 2341
{ 
#line 2342
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
#line 2343
} 
#line 2402
template< class T> static __inline ::cudaError_t 
#line 2403
cudaFuncSetAttribute(T *
#line 2404
func, ::cudaFuncAttribute 
#line 2405
attr, int 
#line 2406
value) 
#line 2408
{ 
#line 2409
return ::cudaFuncSetAttribute((const void *)func, attr, value); 
#line 2410
} 
#line 2435
template< class T> static __inline ::cudaError_t __stdcall 
#line 2436
cudaFuncGetName(const char **
#line 2437
name, T *
#line 2438
func) 
#line 2440
{ 
#line 2441
return ::cudaFuncGetName(name, (const void *)func); 
#line 2442
} 
#line 2458
template< class T> static __inline ::cudaError_t 
#line 2459
cudaGetKernel(::cudaKernel_t *
#line 2460
kernelPtr, T *
#line 2461
func) 
#line 2463
{ 
#line 2464
return ::cudaGetKernel(kernelPtr, (const void *)func); 
#line 2465
} 
#line 2496
template< class T> static __inline ::cudaError_t 
#line 2497
cudaLibraryGetGlobal(T **dptr, ::size_t *bytes, ::cudaLibrary_t library, const char *name) 
#line 2498
{ 
#line 2499
return ::cudaLibraryGetGlobal((void **)((void *)dptr), bytes, library, name); 
#line 2500
} 
#line 2530
template< class T> static __inline ::cudaError_t 
#line 2531
cudaLibraryGetManaged(T **dptr, ::size_t *bytes, ::cudaLibrary_t library, const char *name) 
#line 2532
{ 
#line 2533
return ::cudaLibraryGetManaged((void **)((void *)dptr), bytes, library, name); 
#line 2534
} 
#line 2560
template< class T> static __inline ::cudaError_t 
#line 2561
cudaLibraryGetUnifiedFunction(T **fptr, ::cudaLibrary_t library, const char *symbol) 
#line 2562
{ 
#line 2563
return ::cudaLibraryGetUnifiedFunction((void **)((void *)fptr), library, symbol); 
#line 2564
} 
#line 2579 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/../include\\cuda_runtime.h"
#pragma warning(pop)
#line 64 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
#line 66
const char *info_simulate = ("INFO:simulate[MSVC]"); 
#line 377 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((12 / 10000000) % 10)), (('0') + ((12 / 1000000) % 10)), (('0') + ((12 / 100000) % 10)), (('0') + ((12 / 10000) % 10)), (('0') + ((12 / 1000) % 10)), (('0') + ((12 / 100) % 10)), (('0') + ((12 / 10) % 10)), (('0') + (12 % 10)), '.', (('0') + ((8 / 10000000) % 10)), (('0') + ((8 / 1000000) % 10)), (('0') + ((8 / 100000) % 10)), (('0') + ((8 / 10000) % 10)), (('0') + ((8 / 1000) % 10)), (('0') + ((8 / 100) % 10)), (('0') + ((8 / 10) % 10)), (('0') + (8 % 10)), '.', (('0') + ((61 / 10000000) % 10)), (('0') + ((61 / 1000000) % 10)), (('0') + ((61 / 100000) % 10)), (('0') + ((61 / 10000) % 10)), (('0') + ((61 / 1000) % 10)), (('0') + ((61 / 100) % 10)), (('0') + ((61 / 10) % 10)), (('0') + (61 % 10)), ']', '\000'}; 
#line 406 "CMakeCUDACompilerId.cu"
const char info_simulate_version[] = {'I', 'N', 'F', 'O', ':', 's', 'i', 'm', 'u', 'l', 'a', 't', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + (((1943 / 100) / 10000000) % 10)), (('0') + (((1943 / 100) / 1000000) % 10)), (('0') + (((1943 / 100) / 100000) % 10)), (('0') + (((1943 / 100) / 10000) % 10)), (('0') + (((1943 / 100) / 1000) % 10)), (('0') + (((1943 / 100) / 100) % 10)), (('0') + (((1943 / 100) / 10) % 10)), (('0') + ((1943 / 100) % 10)), '.', (('0') + (((1943 % 100) / 10000000) % 10)), (('0') + (((1943 % 100) / 1000000) % 10)), (('0') + (((1943 % 100) / 100000) % 10)), (('0') + (((1943 % 100) / 10000) % 10)), (('0') + (((1943 % 100) / 1000) % 10)), (('0') + (((1943 % 100) / 100) % 10)), (('0') + (((1943 % 100) / 10) % 10)), (('0') + ((1943 % 100) % 10)), ']', '\000'}; 
#line 426 "CMakeCUDACompilerId.cu"
const char *info_platform = ("INFO:platform[Windows]"); 
#line 427
const char *info_arch = ("INFO:arch[x64]"); 
#line 431
const char *info_language_standard_default = ("INFO:standard_default[03]"); 
#line 447 "CMakeCUDACompilerId.cu"
const char *info_language_extensions_default = ("INFO:extensions_default[OFF]"); 
#line 458 "CMakeCUDACompilerId.cu"
int main(int argc, char *argv[]) 
#line 459
{ 
#line 460
int require = 0; 
#line 461
require += (info_compiler[argc]); 
#line 462
require += (info_platform[argc]); 
#line 464
require += (info_version[argc]); 
#line 467 "CMakeCUDACompilerId.cu"
require += (info_simulate[argc]); 
#line 470 "CMakeCUDACompilerId.cu"
require += (info_simulate_version[argc]); 
#line 472 "CMakeCUDACompilerId.cu"
require += (info_language_standard_default[argc]); 
#line 473
require += (info_language_extensions_default[argc]); 
#line 474
(void)argv; 
#line 475
return require; 
#line 476
} 
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__92f0a630_22_CMakeCUDACompilerId_cu_bd57c623
#ifdef _NV_ANON_NAMESPACE
#endif
#pragma pack()
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
