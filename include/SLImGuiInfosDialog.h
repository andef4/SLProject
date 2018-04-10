//#############################################################################
//  File:      SLImGuiInfosDialog.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLIMGUI_INFOSDIALOG_H
#define SLIMGUI_INFOSDIALOG_H

#include <string>

//-----------------------------------------------------------------------------
//! ImGui UI interface to show specific infos in an imgui dialogue
class SLImGuiInfosDialog
{
public:
    SLImGuiInfosDialog(std::string name)
        : _name(name)
    {
    }
    virtual ~SLImGuiInfosDialog() {}
    virtual void buildInfos() = 0;

    std::string getName() const { return _name; }

private:
    //! name in imgui menu entry for this infos dialogue
    std::string _name;
};

#endif // !SLIMGUI_INFOSDIALOG_H

