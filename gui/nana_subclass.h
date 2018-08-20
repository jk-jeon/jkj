/////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018 Junekey Jeon                                                                   ///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this software     ///
/// and associated documentation files(the "Software"), to deal in the Software without restriction,  ///
/// including without limitation the rights to use, copy, modify, merge, publish, distribute,         ///
/// sublicense, and / or sell copies of the Software, and to permit persons to whom the Software is   ///
/// furnished to do so, subject to the following conditions :                                         ///
///                                                                                                   ///
/// The above copyright notice and this permission notice shall be included in all copies or          ///
/// substantial portions of the Software.                                                             ///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING     ///
/// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND        ///
/// NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,       ///
/// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,    ///
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           ///
/////////////////////////////////////////////////////////////////////////////////////////////////////////

// Minor modifications on
// https://github.com/qPCR4vir/nana-demo/blob/master/Examples/windows-subclassing.cpp

#pragma once
#include <unordered_map>
#include <functional>
#include <mutex>

#include <nana/gui.hpp>

// This header file only works in Windows + MSVC
#include <Windows.h>

class nana_subclass {
	struct msg_pro
	{
		std::function<bool(UINT, WPARAM, LPARAM, LRESULT*)> before;
		std::function<bool(UINT, WPARAM, LPARAM, LRESULT*)> after;
	};

public:
	nana_subclass(nana::window wd) :
		native_{ reinterpret_cast<HWND>(nana::API::root(wd)) },
		old_proc_{ nullptr } {}

	~nana_subclass()
	{
		clear();
	}

	void make_before(UINT msg, std::function<bool(UINT, WPARAM, LPARAM, LRESULT*)> fn)
	{
		if( fn != nullptr ) {
			std::lock_guard<std::mutex> lg{ mutex_ };
			msg_table_[msg].before = std::move(fn);
			_m_subclass(true);
		}
	}

	void make_after(UINT msg, std::function<bool(UINT, WPARAM, LPARAM, LRESULT*)> fn)
	{
		if( fn != nullptr ) {
			std::lock_guard<std::mutex> lg{ mutex_ };
			msg_table_[msg].after = std::move(fn);
			_m_subclass(true);
		}
	}

	void umake_before(UINT msg)
	{
		std::lock_guard<std::mutex> lg{ mutex_ };
		auto i = msg_table_.find(msg);
		if( i != msg_table_.end() )
		{
			i->second.before = nullptr;
			if( i->second.after == nullptr )
			{
				msg_table_.erase(msg);
				if( msg_table_.empty() )
					_m_subclass(false);
			}
		}
	}

	void umake_after(UINT msg)
	{
		std::lock_guard<std::mutex> lg{ mutex_ };
		auto i = msg_table_.find(msg);
		if( i != msg_table_.end() )
		{
			i->second.after = nullptr;
			if( nullptr == i->second.before )
			{
				msg_table_.erase(msg);
				if( msg_table_.empty() )
					_m_subclass(false);
			}
		}
	}

	void umake(UINT msg)
	{
		std::lock_guard<std::mutex> lg{ mutex_ };
		msg_table_.erase(msg);

		if( msg_table_.empty() )
			_m_subclass(false);
	}

	void clear()
	{
		std::lock_guard<std::mutex> lg{ mutex_ };
		msg_table_.clear();
		_m_subclass(false);
	}

private:
	void _m_subclass(bool enable)
	{
		if( enable ) {
			if( native_ && (nullptr == old_proc_) ) {
				old_proc_ = (WNDPROC)::SetWindowLongPtr(native_, GWLP_WNDPROC, (LONG_PTR)_m_subclass_proc);
				if( old_proc_ )
					table_[native_] = this;
			}
		}
		else {
			if( old_proc_ ) {
				table_.erase(native_);
				::SetWindowLongPtr(native_, GWLP_WNDPROC, (LONG_PTR)old_proc_);
				old_proc_ = nullptr;

			}
		}
	}

	static bool _m_call_before(msg_pro& pro, UINT msg, WPARAM wp, LPARAM lp, LRESULT* res)
	{
		return (pro.before ? pro.before(msg, wp, lp, res) : true);
	}

	static bool _m_call_after(msg_pro& pro, UINT msg, WPARAM wp, LPARAM lp, LRESULT* res)
	{
		return (pro.after ? pro.after(msg, wp, lp, res) : true);
	}

private:
	static LRESULT CALLBACK _m_subclass_proc(HWND wd, UINT msg, WPARAM wp, LPARAM lp)
	{
		std::lock_guard<std::mutex> lg{ mutex_ };

		nana_subclass* self = _m_find(wd);
		if( self == nullptr || self->old_proc_ == nullptr )
			return ::DefWindowProc(wd, msg, wp, lp);

		auto i = self->msg_table_.find(msg);
		if( i == self->msg_table_.end() )
			return ::CallWindowProc(self->old_proc_, wd, msg, wp, lp);

		LRESULT res = 0;
		if( self->_m_call_before(i->second, msg, wp, lp, &res) )
		{
			res = ::CallWindowProc(self->old_proc_, wd, msg, wp, lp);
			self->_m_call_after(i->second, msg, wp, lp, &res);
		}

		if( WM_DESTROY == msg )
			self->clear();

		return res;
	}

	static nana_subclass* _m_find(HWND wd)
	{
		auto i = table_.find(wd);
		if( i != table_.end() )
			return i->second;
		return nullptr;
	}

private:
	HWND native_;
	WNDPROC old_proc_;
	std::unordered_map<UINT, msg_pro> msg_table_;

	static std::mutex mutex_;
	static std::unordered_map<HWND, nana_subclass*> table_;
};

std::mutex nana_subclass::mutex_;
std::unordered_map<HWND, nana_subclass*> nana_subclass::table_;