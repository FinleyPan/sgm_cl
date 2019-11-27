/**
 * definitions for inline and template function members
 */

template <typename... Types>
void CLKernel::SetArgs(Types&&... args)
{
    size_t num_args = sizeof...(args);
    void* arguments[sizeof...(args)];
    size_t args_sizeof[sizeof...(args)];
    FillArgVector(0, arguments, args_sizeof, args...);
    SetArgs(int(num_args), arguments, args_sizeof);
}

inline void CLKernel::FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof)
{}

template <typename... Types>
void CLKernel::FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof,
                                                CLBuffer* arg, Types&&... Fargs){
   auto prop = arg->GetArgumentPropereties();
   arg_address[arg_idx] = prop.arg_ptr;
   arg_sizeof[arg_idx] = prop.sizeof_arg;
   FillArgVector(++arg_idx, arg_address, arg_sizeof, Fargs...);
}

template <typename T, typename... Types>
void CLKernel::FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof,
                                                      T&& arg, Types&&... Fargs){
    arg_address[arg_idx] = &arg;
    arg_sizeof[arg_idx] = sizeof(arg);
    FillArgVector(++arg_idx, arg_address, arg_sizeof, Fargs...);
}

