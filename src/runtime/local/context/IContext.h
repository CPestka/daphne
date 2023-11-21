#pragma once

class IContext {
public:
    virtual void destroy() = 0;
    virtual ~IContext() = default;
};
