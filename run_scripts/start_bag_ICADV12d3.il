;; SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
;; Copyright 2018 Regents of the University of California
;; All rights reserved.
;;
;; Redistribution and use in source and binary forms, with or without
;; modification, are permitted provided that the following conditions are met:
;;
;; * Redistributions of source code must retain the above copyright notice, this
;;   list of conditions and the following disclaimer.
;;
;; * Redistributions in binary form must reproduce the above copyright notice,
;;   this list of conditions and the following disclaimer in the documentation
;;   and/or other materials provided with the distribution.
;;
;; * Neither the name of the copyright holder nor the names of its
;;   contributors may be used to endorse or promote products derived from
;;   this software without specific prior written permission.
;;
;; THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
;; AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
;; IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
;; DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
;; FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
;; DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
;; SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
;; CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
;; OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
;; OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

;; Copyright 2019 Blue Cheetah Analog Design Inc.
;;
;; Licensed under the Apache License, Version 2.0 (the "License");
;; you may not use this file except in compliance with the License.
;; You may obtain a copy of the License at
;;
;;     http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.

/*  Note:

Due to licensing reasons, this skill script is missing the function
CCSinvokeCdfCallbacks() from Cadence solution 11018344, which executes
CDF parameters callback from skill.

If you do not need to instantiate a pcell instance, this method
is not needed.

Eric Chang, Mar 2, 2017.

*/


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Virtuoso Database operations functions  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; reads a skill data structure from file
procedure( parse_data_from_file( fname "t" )
    let( (p ans)
        unless( p = infile( fname )
            error("Cannot open file %s" fname)
        )
        ans = parse_data_from_file_helper(p)
        close( p )
        ans
    )
)

; recursive helper for parse_data_from_file
procedure( parse_data_from_file_helper( p )
    let( (line item ans finish key)
        gets( line p )
        ; remove newline
        line = substring(line 1 strlen(line) - 1)
        ; printf("read line: %s\n" line)
        cond(
            (line == "#list"
            ; parse a list
            ans = tconc(nil 0)
            while( nequal(item = parse_data_from_file_helper(p) "#end")
                tconc(ans item)
            )
            ; printf("returning list ")
            ; print(cdar(ans))
            ; printf("\n")
            cdar(ans)
            )
            (line == "#prop_list"
            ; parse a disembodied property list
            ans = ncons(nil)
            finish = nil
            while( !finish
                key = parse_data_from_file_helper(p)
                if( key == "#end" then
                    finish = 't
                else
                    item = parse_data_from_file_helper(p)
                    putprop(ans item key)
                )
            )
            ans
            )
            ; parse a float
            (strncmp( line "#float" 6 ) == 0
                cdfParseFloatString(cadr(parseString(line)))
            )
            ; parse an int
            (strncmp( line "#int" 4 ) == 0
                atoi(cadr(parseString(line)))
            )
            ; parse a boolean
            (strncmp( line "#bool" 5 ) == 0
                if( atoi(cadr(parseString(line))) == 1 then
                    't
                else
                    nil
                )
            )
            ; parse a string token or #end
            ('t
                ; printf("returning str %s\n" line)
                line
            )
        )
    )
)

; close all opened cellviews
procedure( close_all_cellviews()
    foreach( cv dbGetOpenCellViews()
        dbPurge(cv)
    )
    't
)

; release write locks from all the given cellviews
procedure( release_write_locks( lib_name cell_view_list_f "tt" )
    let( (cell_view_list lib_obj cv)
        cell_view_list = parse_data_from_file(cell_view_list_f)
        when( lib_obj = ddGetObj(lib_name nil nil nil nil "r")
            foreach( info cell_view_list
                when( cv = dbFindOpenCellView( lib_obj car(info) cadr(info) )
                    dbReopen(cv, "r")
                    dbClose(cv)
                )
            )
            ddReleaseObj(lib_obj)
        )
        t
    )
)

; refresh all given cell views
procedure( refresh_cellviews( lib_name cell_view_list_f "tt" )
    let( (cell_view_list lib_obj cv)
        cell_view_list = parse_data_from_file(cell_view_list_f)
        when( lib_obj = ddGetObj(lib_name nil nil nil nil "r")
            foreach( info cell_view_list
                when( cv = dbFindOpenCellView( lib_obj car(info) cadr(info) )
                    dbRefreshCellView(cv)
                )
            )
            ddReleaseObj(lib_obj)
        )
        t
    )
)

; if library with lib_name does not exist, create a new library with that name.
; Otherwise, if erase is true, remove all cells in that library.
; Returns the library database object.
procedure( create_or_erase_library(lib_name tech_lib lib_path erase "tttg")
    let( (lib_obj)
        if( lib_obj = ddGetObj(lib_name nil nil nil nil "r") then
            when( erase
                ; delete all cells in the library
                foreach( cell lib_obj~>cells
                    unless( ddDeleteObj(cell)
                        error("cannot delete cell %s in library %s\n" cell~>name lib_name)
                    )
                )
            )
            ddReleaseObj(lib_obj)
            't
        else
            ; create library if not exist
            when( and(lib_path (lib_path != "."))
                lib_path = strcat(lib_path "/" lib_name)
            )
            lib_obj = ddCreateLib(lib_name lib_path)
            ; attach technology file
            techBindTechFile(lib_obj tech_lib)
            ; close library
            ddReleaseObj(lib_obj)
            't
        )
    )
)

; modify label text in a layout
procedure( modify_label(lib_name cell_name reg_before reg_after "tttt")
    let((cv lbl_txt new_lbl)
        unless( cv = dbOpenCellViewByType( lib_name cell_name "layout" nil "a" )
            error( "Cannot find layout view of cell %s__%s" lib_name cell_name )
        )

        foreach(shape_obj cv~>shapes
            when( lbl_txt = shape_obj->theLabel
                when( rexMatchp(reg_before lbl_txt)
                    new_lbl = rexReplace(lbl_txt reg_after 1)
                    shape_obj->theLabel = new_lbl
                )
            )
        )

        dbSave(cv)
        dbClose(cv)
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; The following functions import Virtuoso schematics and symbols into yamls
;; and a python skeleton for the schematic generator.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; This function is used to print the bounding box in parse_cad_sch()
procedure( print_bbox(p indent bbox dbu_per_uu)
    fprintf( p "%sbbox:\n" indent )
    foreach( pair bbox
        foreach( elem pair
            fprintf( p "%s  - %d\n" indent round(elem * dbu_per_uu) )
        )
    )
)

; This function is used to print the signal enum in parse_cad_sch()
procedure( print_sig_type(p indent sig_type)
    let( (sig_enum)
        if(sig_type == "signal" then
            sig_enum = 0
        else if(sig_type == "supply" then
            sig_enum = 1
        else    ; sig_type == "ground"
            sig_enum = 2
        ))
        fprintf( p "%sstype: %d\n" indent sig_enum )
    )
)

; This function is used to print the term enum using direction in parse_cad_sch()
procedure( print_term_type_from_direction(p indent direction)
    let( (term_enum)
        if(direction == "inputOutput" then
            term_enum = 2
        else if(direction == "input" then
            term_enum = 0
        else    ; direction == "output"
            term_enum = 1
        ))
        fprintf( p "%sttype: %d\n" indent term_enum )
    )
)

; This function is used to print the term enum using name in parse_cad_sch()
procedure( print_term_type_from_name(p indent name)
    let( (term_enum)
        if(name == "iopin" then
            term_enum = 2
        else if(name == "ipin" then
            term_enum = 0
        else    ; name == "opin"
            term_enum = 1
        ))
        fprintf( p "%sttype: %d\n" indent term_enum )
    )
)

; This function is used to print the transformation in parse_cad_sch()
procedure( print_xform(p indent xform dbu_per_uu)
    let( (xform_idx)
        fprintf( p "%sxform:\n" indent )
        xform_idx = 0
        foreach( xf xform
            if(xform_idx == 1 then
                fprintf( p "%s  - %s\n" indent xf)
            else if(xform_idx == 0 then
                foreach( xy xf
                    fprintf( p "%s  - %d\n" indent round(xy * dbu_per_uu))
                )
            ))
            xform_idx++
        )
    )
)

; This function is used to print "is_primitive" in parse_cad_sch()
procedure( print_is_primitive(p indent lib_name)
    let( (is_primitive)
        if( lib_name == "basic" || lib_name == "analogLib" || lib_name == "ahdlLib" || lib_name == "BAG_prim" then
            is_primitive = "true"
        else
            is_primitive = "false"
        )
        fprintf( p "%sis_primitive: %s\n" indent is_primitive )
    )
)

; This function is used to print the origin in parse_cad_sch()
procedure( print_origin(p indent origin dbu_per_uu)
    fprintf( p "%sorigin:\n" indent )
    foreach( xy origin
        fprintf( p "%s  - %d\n" indent round(xy * dbu_per_uu) )
    )
)

; This function is used to print the alignment enum in parse_cad_sch()
procedure( print_alignment(p indent align_type)
    let( (align_enum)
        if( align_type == "centerRight" then
            align_enum = 7
        else if( align_type == "centerLeft" then
            align_enum = 1
        else if( align_type == "centerCenter" then
            align_enum = 4
        else if( align_type == "lowerLeft" then
            align_enum = 2
        else if( align_type == "lowerRight" then
            align_enum = 8
        else if( align_type == "lowerCenter" then
            align_enum = 5
        else if( align_type == "upperLeft" then
            align_enum = 0
        else if( align_type == "upperRight" then
            align_enum = 6
        else if( align_type == "upperCenter" then
            align_enum = 3
        else
            error( "Unrecognized justify = %s" align_type )
        )))))))))
        fprintf( p "%salignment: %d\n" indent align_enum )
    )
)

; This function is used to print the font enum in parse_cad_sch()
procedure( print_font(p indent font_type)
    let( (font_enum)
        if( font_type == "stick" then
            font_enum = 5
        else if( font_type == "roman" then
            font_enum = 3
        else
            error( "Unrecognized font = %s" font_type )
        ))
        fprintf( p "%sfont: %d\n" indent font_enum )
    )
)

; This function is used to print boolean values in parse_cad_sch()
procedure( print_bool(p indent attr attr_name)
    if( attr then
        fprintf( p "%s%s: true\n" indent attr_name )
    else
        fprintf( p "%s%s: false\n" indent attr_name )
    )
)

defMathConstants('MyConsts)
; Parse the netlist of the given cellview.
; Works on schematic and veriloga and symbol.
procedure( parse_cad_sch(lib_name cell_name view_name file_name "tttt")
    let( (skip cv p indent shp_status shp_ptr inst term_cnt inst_cnt prop_cnt obj_type purp_type path_enum)
        skip = 0
        if(view_name == "schematic" then
            unless( cv = dbOpenCellViewByType( lib_name cell_name view_name nil "r" )
                view_name = "veriloga"
                unless( cv = dbOpenCellViewByType( lib_name cell_name view_name nil "r" )
                    error( "Cannot find schematic or veriloga view of cell %s__%s" lib_name cell_name )
                )
            )
        else
            unless( cv = dbOpenCellViewByType( lib_name cell_name view_name nil "r" )
                skip = 1
            )
        )
        if(skip == 0 then
            indent = ""
            p = outfile( file_name "w" )

            ; print cellview information
            printf( "*INFO* Writing cell %s__%s (%s) netlist to %s\n" lib_name cell_name view_name file_name )
            fprintf( p "%slib_name: %s\n" indent lib_name )
            fprintf( p "%scell_name: %s\n" indent cell_name )
            fprintf( p "%sview_name: %s\n" indent view_name )

            print_bbox(p indent cv~>bBox cv~>DBUPerUU)

            ; print terminals
            indent = ""
            term_cnt = 0
            fprintf( p "%sterminals:\n" indent )
            if(view_name == "symbol" then
                ; use cv~>shapes instead of cv~>terminals since the first has more information
                foreach( shape cv~>shapes
                    ; pick the shape only if it is (pin, drawing)
                    if(shape~>layerNum == 229 && shape~>purpose == "drawing" then
                        term_cnt++

                        ; print entry for terminal
                        indent = "  "
                        fprintf( p "%s%s:\n" indent shape~>net~>name )
                        ; print terminal information
                        indent = "    "
                        fprintf( p "%sobj:\n" indent )
                        indent = "      "
                        fprintf( p "%s- 0\n" indent )   ; enum for symbol terminals
                        fprintf( p "%s- layer: %d\n" indent shape~>layerNum )
                        fprintf( p "%s  purpose: 4294967295\n" indent )     ; get purpose number from cv?
                        fprintf( p "%s  net: \"\"\n" indent )

                        indent = "        "
                        print_bbox(p indent shape~>bBox cv~>DBUPerUU)

                        indent = "    "
                        print_sig_type(p indent shape~>net~>sigType)
                        print_term_type_from_direction(p indent shape~>pin~>term~>direction)
                    )
                )
            else
                ; use cv~>shapes instead of cv~>terminals since the first has more information
                foreach( shape cv~>shapes
                    if(shape~>layerNum == 229 then
                        term_cnt++

                        ; print entry for terminal
                        indent = "  "
                        fprintf( p "%s%s:\n" indent shape~>associate~>name )
                        ; print terminal information
                        indent = "    "
                        fprintf( p "%sobj:\n" indent )
                        indent = "      "
                        fprintf( p "%s- 1\n" indent )   ; enum for schematic terminals
                        fprintf( p "%s- inst:\n" indent )

                        ; find corresponding instance in cv~>instances, and exit once match is found
                        shp_status = t
                        shp_ptr = cv~>instances
                        while(shp_status && shp_ptr
                            inst = car(shp_ptr)
                            if(inst~>net~>name == shape~>associate~>name then
                                ; found match, exit
                                shp_status = nil
                            else
                                shp_ptr = cdr(shp_ptr)
                            )
                        )

                        indent = "          "
                        fprintf( p "%slib_name: %s\n" indent inst~>libName )
                        fprintf( p "%scell_name: %s\n" indent inst~>cellName )
                        fprintf( p "%sview_name: %s\n" indent inst~>viewName )

                        print_xform(p indent inst~>transform cv~>DBUPerUU)
                        print_bbox(p indent inst~>bBox cv~>DBUPerUU)

                        fprintf( p "%sconnections:\n" indent )
                        fprintf( p "%s  {}\n" indent )
                        fprintf( p "%sparams:\n" indent )
                        fprintf( p "%s  {}\n" indent )

                        print_is_primitive(p indent inst~>libName)

                        ; print attr
                        indent = "      "
                        fprintf( p "%s  attr:\n" indent )
                        indent = "          "
                        fprintf( p "%slayer: %d\n" indent shape~>layerNum )
                        fprintf( p "%spurpose: 237\n" indent )  ; get purpose number from cv?
                        fprintf( p "%snet: \"\"\n" indent )
                        print_origin(p indent shape~>xy cv~>DBUPerUU)

                        print_alignment(p indent shape~>justify)

                        fprintf( p "%sorient: %s\n" indent shape~>orient )
                        print_font(p indent shape~>font)
                        fprintf( p "%sheight: %d\n" indent round(shape~>height * cv~>DBUPerUU) )

                        print_bool(p indent shape~>isOverbar "overbar")
                        print_bool(p indent shape~>isVisible "visible")
                        print_bool(p indent shape~>isDrafted "drafting")
                        if( shape~>assocType == "attrAssocType" then
                            fprintf( p "%sattr_type: 0\n" indent )
                        else
                            error( "Unrecognized assocType = %s" shape~>assocType )
                        )
                        fprintf( p "%sformat: 1\n" indent )     ; get from cv?

                        indent = "    "
                        print_sig_type(p indent inst~>net~>sigType)
                        print_term_type_from_name(p indent inst~>cellName)
                    )
                )
            )
            when(term_cnt == 0
                fprintf( p "  {}\n" )
            )

            indent = ""
            fprintf( p "%sshapes:\n" indent )
            foreach( shape cv~>shapes
                ; if schematic, pick only if it's not pin
                ; if symbol, pick only if it's not (pin, drawing)
                if( (view_name == "schematic" && shape~>layerNum != 229) || (view_name == "symbol" && !(shape~>layerNum == 229 && shape~>purpose == "drawing")) then
                    indent = "  "
                    fprintf( p "%s-\n" indent )

                    indent = "    "
                    if( shape~>objType == "label"  then
                        if( shape~>layerName == "wire" || shape~>layerName == "pin" then
                            obj_type = 7   ; enum for wire label and pin label
                        else if( shape~>layerName == "device" || shape~>layerName == "instance" then
                            if( shape~>purpose == "annotate" then
                                obj_type = 7   ; enum for device annotate
                            else
                                obj_type = 8   ; enum for device label and instance label
                            )
                        else
                            error( "Unrecognized layerName = %s" shape~>layerName )
                        ))
                        fprintf( p "%s- %d\n" indent obj_type )
                        fprintf( p "%s- layer: %d\n" indent shape~>layerNum )

                        if( shape~>purpose == "label" then
                            purp_type = 237    ; get purpose number from cv?
                        else if( shape~>purpose == "annotate" then
                            purp_type = 240    ; get purpose number from cv?
                        else
                            error( "Unrecognized purpose = %s" shape~>purpose )
                        ))
                        fprintf( p "%s  purpose: %d\n" indent purp_type )

                        indent = "      "
                        if( shape~>net~>name then
                            fprintf( p "%snet: %s\n" indent shape~>net~>name )
                        else
                            fprintf( p "%snet: \"\"\n" indent )
                        )
                        print_origin(p indent shape~>xy cv~>DBUPerUU)
                        print_alignment(p indent shape~>justify)

                        fprintf( p "%sorient: %s\n" indent shape~>orient )
                        print_font(p indent shape~>font)
                        fprintf( p "%sheight: %d\n" indent round(shape~>height * cv~>DBUPerUU) )
                        print_bool(p indent shape~>isOverbar "overbar")
                        print_bool(p indent shape~>isVisible "visible")
                        print_bool(p indent shape~>isDrafted "drafting")

                        if( shape~>layerName == "device" || shape~>layerName == "instance" then
                            if( shape~>purpose == "label" then
                                fprintf( p "%stext: \"%s\"\n" indent shape~>theLabel )
                                fprintf( p "%sevaluator: cdsNLPEvalText\n" indent )   ; get from cv?
                            else
                                fprintf( p "%stext: %s\n" indent shape~>theLabel )
                            )
                        else
                            if(shape~>theLabel then
                                fprintf( p "%stext: %s\n" indent shape~>theLabel )
                            else
                                fprintf( p "%stext: %s\n" indent shape~>net~>name )
                            )
                        )
                    else
                        if( shape~>objType == "line" then
                            obj_type = 5   ; enum for wire drawing and some device drawing
                        else if( shape~>objType == "path" then
                            obj_type = 6   ; enum for some device drawing1
                        else if( shape~>objType == "arc" then
                            obj_type = 2   ; enum for some device drawing
                        else if( shape~>objType == "rect" then
                            obj_type = 0   ; enum for instance drawing and some device drawing
                        else if( shape~>objType == "ellipse" then
                            obj_type = 4   ; enum for some device drawing1 and some device drawing
                        else
                            error( "Unrecognized objType = %s" shape~>objType )
                        )))))
                        fprintf( p "%s- %d\n" indent obj_type )
                        fprintf( p "%s- layer: %d\n" indent shape~>layerNum )
                        if( shape~>purpose == "drawing" then
                            fprintf( p "%s  purpose: 4294967295\n" indent )     ; get purpose number from cv?
                        else if( shape~>purpose == "drawing1" then
                            fprintf( p "%s  purpose: 241\n" indent )    ; get purpose number from cv?
                        else
                            error( "Unrecognized purpose = %s" shape~>purpose )
                        ))
                        if( shape~>net then
                            fprintf( p "%s  net: %s\n" indent shape~>net~>name )
                        else
                            fprintf( p "%s  net: \"\"\n" indent )
                        )

                        if( shape~>objType == "line" || shape~>objType == "path" then
                            if( shape~>objType == "path" then
                                fprintf( p "%s  width: %d\n" indent round(shape~>width * cv~>DBUPerUU) )
                            )
                            fprintf( p "%s  points:\n" indent )
                            foreach( pair shape~>points
                                fprintf( p "%s    -\n" indent )
                                foreach( elem pair
                                    fprintf( p "%s      - %d\n" indent round(elem * cv~>DBUPerUU) )
                                )
                            )
                            if( shape~>objType == "path" then
                                if( shape~>pathStyle == "roundRound" then
                                    path_enum = 2    ; enum for round path style
                                else
                                    error( "Unrecognized path style = %s" shape~>pathStyle )
                                )
                                fprintf( p "%s  style: %d\n" indent path_enum )
                                fprintf( p "%s  begin_ext: %d\n" indent round(shape~>beginExt * cv~>DBUPerUU) )
                                fprintf( p "%s  end_ext: %d\n" indent round(shape~>endExt * cv~>DBUPerUU) )
                            )
                        else
                            if( shape~>objType == "arc" then
                                if( shape~>startAngle < 0 then
                                    fprintf( p "%s  ang_start: %.16f\n" indent 2 * MyConsts.PI + shape~>startAngle )
                                else
                                    fprintf( p "%s  ang_start: %.16f\n" indent shape~>startAngle )
                                )
                                if( shape~>stopAngle < 0 then
                                    fprintf( p "%s  ang_stop: %.16f\n" indent 2 * MyConsts.PI + shape~>stopAngle )
                                else
                                    fprintf( p "%s  ang_stop: %.16f\n" indent shape~>stopAngle )
                                )
                            )
                            indent = "      "
                            print_bbox(p indent shape~>bBox cv~>DBUPerUU)
                        )
                    )
                )
            )

            ; print instances
            indent = ""
            inst_cnt = 0
            fprintf( p "%sinstances:\n" indent )
            foreach( inst cv~>instances
                if(inst~>purpose != "pin" then
                    inst_cnt++

                    ; print entry for instance
                    indent = "  "
                    fprintf( p "%s%s:\n" indent inst~>name )
                    ; print instance master information.
                    indent = "    "
                    fprintf( p "%slib_name: %s\n" indent inst~>libName )
                    fprintf( p "%scell_name: %s\n" indent inst~>cellName )
                    fprintf( p "%sview_name: %s\n" indent inst~>viewName )

                    print_xform(p indent inst~>transform cv~>DBUPerUU)
                    print_bbox(p indent inst~>bBox cv~>DBUPerUU)

                    ; print instance terminal information
                    if( !(inst~>instTerms) then
                        fprintf( p "%sconnections: {}\n" indent )
                    else
                        fprintf( p "%sconnections:\n" indent )
                        foreach( inst_term inst~>instTerms
                            indent = "      "
                            fprintf( p "%s%s: %s\n" indent inst_term~>name inst_term~>net~>name )
                        )
                    )
                    indent = "    "

                    ; print params
                    fprintf( p "%sparams:\n" indent )
                    prop_cnt = 0
                    foreach(param inst~>prop
                        fprintf( p "%s  %s:\n" indent param~>name )
                        fprintf( p "%s    - 3\n" indent ) ; enum for string
                        if(param~>value == "" then
                            fprintf( p "%s    - \"\"\n" indent )
                        else
                            fprintf( p "%s    - %s\n" indent param~>value )
                        )
                        prop_cnt++
                    )
                    when(prop_cnt == 0
                        fprintf( p "%s  {}\n" indent )
                    )

                    print_is_primitive(p indent inst~>libName)
                )
            )
            when(inst_cnt == 0
                fprintf( p "  {}\n" )
            )

            ; print properties
            indent = ""
            fprintf( p "%sprops:\n" indent )
            foreach( _prop cv~>prop
                indent = "  "
                fprintf( p "%s%s:\n" indent _prop~>name )
                indent = "    "
                if( _prop~>valueType == "string" then
                    fprintf( p "%s- 3\n" indent )   ; enum for string
                    if( _prop~>value == "" then
                        fprintf( p "%s- \"\"\n" indent )
                    else
                        fprintf( p "%s- %s\n" indent _prop~>value )
                    )
                else if( _prop~>valueType == "int" then
                    fprintf( p "%s- 0\n" indent )   ; enum for int
                    fprintf( p "%s- %d\n" indent _prop~>value )
                else if( _prop~>valueType == "time" then
                    fprintf( p "%s- 4\n" indent )   ; enum for time
                    fprintf( p "%s- time_val: %d\n" indent stringToTime(_prop~>value) )
                else if( _prop~>valueType == "ILList" then
                    fprintf( p "%s- 5\n" indent )   ; enum for ILList
                    fprintf( p "%s- name: ILList\n" indent )
                    fprintf( p "%s  bin_val: (" indent )
                    prop_cnt = 0
                    foreach( _elem _prop~>value
                        if( prop_cnt > 0
                            fprintf( p " " )
                        )
                        fprintf( p "\"%s\"" _elem )
                        prop_cnt++
                    )
                    fprintf( p ")\n" )
                ))))
            )

            ; print app_defs
            indent = ""
            fprintf( p "%sapp_defs:\n" indent )
            fprintf( p "%s  _dbLastSavedCounter:\n" indent )
            fprintf( p "%s    - 0\n" indent )
            fprintf( p "%s    - %d\n" indent cv~>modifiedCounter )
            fprintf( p "%s  _dbvCvTimeStamp:\n" indent )
            fprintf( p "%s    - 0\n" indent )
            fprintf( p "%s    - %d\n" indent cv~>modifiedCounter )
            fprintf( p "%s  cdbRevision:\n" indent )
            fprintf( p "%s    - 0\n" indent )
            fprintf( p "%s    - 227612\n" indent )  ; get from cv?
            ; TODO: get cdnSPDesignMajorVersion

            ; close resources
            close(p)
            dbClose(cv)
        )
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; The following functions export Virtuoso schematics and symbols.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; copy all template cells to the given library.
; template list is a list of three-element lists with the format
; '("master_lib_name" "master_cell_name" "target_cell_name")
; any existing cellviews will be overwritten.
procedure( copy_templates_to_library(lib_name template_list "tl")
    let( (current remaining src_gdm targ_gdm table master_lib master_cell target_cell key cnt
          empty_spec targ_lib_obj test_cv)

        current = template_list
        remaining = '()
        empty_spec = gdmCreateSpecList()
        targ_lib_obj = ddGetObj(lib_name nil nil nil nil "r")

        ; ccpCopy cannot copy the same cell to multiple different cells.
        ; because of this, we need to copy a set of unique cells at a time,
        ; hence the while loop.
        while( current
            ; Create GDMSpecList used to copy all cells
            src_gdm = gdmCreateSpecList()
            targ_gdm = gdmCreateSpecList()
            ; table to keep track of seen cells.
            table = makeTable("mytable" 0)
            ; Populate GDMSpecList
            foreach( template_info current
                master_lib = car(template_info)
                master_cell = cadr(template_info)
                target_cell = caddr(template_info)

                ; check if we copied this cell on this iteration yet
                key = list(master_lib master_cell)
                if( table[key] == 1 then
                    ; wait for the next iteration
                    remaining = cons(template_info remaining)
                else
                    ; purge target cellview if exist
                    when( targ_lib_obj
                        test_cv = dbFindOpenCellView(targ_lib_obj target_cell "schematic")
                        when( test_cv
                            dbPurge(test_cv)
                        )
                        test_cv = dbFindOpenCellView(targ_lib_obj target_cell "symbol")
                        when( test_cv
                            dbPurge(test_cv)
                        )
                        ; hard remove adexl state if it exists
                        test_cv = ddGetObj(lib_name target_cell "adexl")
                        when( test_cv
                            ddDeleteObj(test_cv)
                        )
                    )
                    gdmAddSpecToSpecList(gdmCreateSpec(master_lib master_cell nil nil "CDBA") src_gdm)
                    gdmAddSpecToSpecList(gdmCreateSpec(lib_name target_cell nil nil "CDBA") targ_gdm)
                    table[key] = 1
                )
            )
            ; Perform copy
            ccpCopy(src_gdm targ_gdm 't 'CCP_EXPAND_COMANAGED nil nil "" "" 'CCP_UPDATE_FROM_LIBLIST empty_spec)

            ; set current and remaining
            current = remaining
            remaining = '()

            ; debug printing
            ; printstruct(table)
        )
    )
    't
)

; helper method to open pin master
procedure( open_pin_master(cvid pin_cv_info)
    let( (pin_master mpin_lib mpin_cell mpin_view)
        mpin_lib = car(pin_cv_info)
        mpin_cell = cadr(pin_cv_info)
        mpin_view = caddr(pin_cv_info)
        unless( pin_master = dbOpenCellViewByType( mpin_lib mpin_cell mpin_view nil "r" )
            dbClose(cvid)
            error( "Cannot find pin master cellview: %s__%s (%s)" mpin_lib mpin_cell mpin_view)
        )
        pin_master
    )
)

; find X and Y coordinates to insert new symbol pins
procedure( get_new_pin_locations(cvid snap_dist)
    let( (pin bbox pin_x pin_y xl xr yl yr)
        ; find the left-most/right-most pin X coordinates, and find the lowst
        ; Y coordinate of the left-most/right-most pins
        xl = nil
        xr = nil
        yl = nil
        yr = nil
        foreach( term cvid->terminals
            when( term~>pinCount != 1
                dbClose(cvid)
                error( "Terminal %s does not have exactly one pin" term~>name)
            )
            pin = car(term~>pins)
            bbox = pin~>fig~>bBox
            pin_x = round2((xCoord(car(bbox)) + xCoord(cadr(bbox))) / 2.0 / snap_dist)
            pin_y = round2((yCoord(car(bbox)) + yCoord(cadr(bbox))) / 2.0 / snap_dist)
            if( xl == nil then
                xl = pin_x
                xr = pin_x
                yl = pin_y
                yr = pin_y
            else
                cond( (pin_x < xl xl = pin_x yl = pin_y)
                      (pin_x == xl yl = min(yl pin_y)))
                cond( (pin_x > xr xr = pin_x yr = pin_y)
                      (pin_x == xr yr = min(yr pin_y)))
            )
        )
        when(xl == nil
            ; default values if schematic has no terminals
            ; this usually means you have a testbench schematic
            xl = 0
            yl = 0
            xr = 10
            yr = 0
        )
        list((xl * snap_dist):(yl * snap_dist) (xr * snap_dist):(yr * snap_dist))
    )
)

; returns a unique terminal name in the given cellview.
; name_base is the suffix of the returned terminal name.
procedure( get_unique_term_name( cvid name_base "gt")
    let( (cnt new_term_name)
        cnt = 1
        sprintf( new_term_name "temp%d_%s" cnt name_base )
        while( dbFindTermByName(cvid new_term_name)
            cnt = cnt + 1
            sprintf( new_term_name "temp%d_%s" cnt name_base )
        )
        new_term_name
    )
)

; update pins of a symbol
; pin_map is a list of two-element lists, first element is old pin name, second element is new pin name.
; sympin is a 3-element list of strings. first element is the pin master library,
; second element is the pin master cell, and third element is the pin master cellview.
; simulators is a list of simulator names for which termOrder should be updated.
; Usually simulators = '("auLvs" "auCdl" "spectre" "hspiceD")
procedure( update_symbol_pin(lib_name cell_name pin_map new_pins sympin simulators "ttllll")
    let( (snap_dist cvid pin_master cur_term_name new_term_name term pin bbox pin_x pin_y pin_location pin_direction
          label_location label_rel_location temp_new_term_name new_term new_port_order cell_obj bc
          mpin_lib mpin_cell mpin_view pin_xy_info npin_xl npin_yl npin_xr npin_yr npin_name npin_type
          modified_pins)

        snap_dist = schGetEnv("schSnapSpacing")
        modified_pins = nil
        mpin_lib = car(sympin)
        mpin_cell = cadr(sympin)
        mpin_view = caddr(sympin)
        unless( pin_master = dbOpenCellViewByType(mpin_lib mpin_cell mpin_view nil "r")
            error("Cannot open symbol pin cellview %s__%s (%s)." mpin_lib mpin_cell mpin_view)
        )
        unless( cvid = dbOpenCellViewByType(lib_name cell_name "symbol" nil "a")
            dbClose(pin_master)
            error("Cannot open cellview %s__%s (symbol)." lib_name cell_name)
        )

        ; get new pin locations before any pin addition/substraction.
        pin_xy_info = get_new_pin_locations(cvid snap_dist)

        ; modify existing pins
        new_port_order = tconc(nil "")
        foreach( p pin_map
            cur_term_name = car(p)
            new_term_name = cadr(p)
            new_port_order = tconc(new_port_order new_term_name)
            when( cur_term_name != new_term_name
                modified_pins = 't
                ; printf("%s %s\n" cur_term_name new_term_name)
                unless( term = dbFindTermByName(cvid cur_term_name)
                    dbClose(pin_master)
                    dbReopen(cvid, "r")
                    dbClose(cvid)
                    error( "Terminal %s not found." cur_term_name )
                )
                when( term~>pinCount != 1
                    dbClose(pin_master)
                    dbReopen(cvid, "r")
                    dbClose(cvid)
                    error( "Terminal %s does not have exactly one pin." cur_term_name)
                )
                pin = car(term~>pins)

                if( strlen(new_term_name) != 0 then
                    ; rename pin
                    bbox = pin~>fig~>bBox
                    pin_x = (xCoord(car(bbox)) + xCoord(cadr(bbox))) / 2.0
                    pin_y = (yCoord(car(bbox)) + yCoord(cadr(bbox))) / 2.0
                    pin_location = round2(pin_x / snap_dist) * snap_dist:round2(pin_y / snap_dist) * snap_dist
                    pin_direction = term~>direction

                    ; change label
                    prog( (label_orientation label_font label_font_size label_type label_text)
                        foreach( label pin~>fig~>children
                            when( label~>objType == "label"
                                label_location = label~>xy
                                label_orientation = label~>orient
                                label_rel_location = label~>justify
                                label_font = label~>font
                                label_font_size = label~>height
                                label_type = label~>labelType
                                label_text = label~>theLabel
                                when( label_text == cur_term_name
                                    schCreateSymbolLabel(cvid label_location "pin label" new_term_name label_rel_location
                                                         label_orientation label_font label_font_size label_type)
                                    return('t)
                                )
                            )
                        )
                        return(nil)
                    )

                    dbDeleteObject(pin~>fig)
                    dbDeleteObject(pin)

                    ;create a temporary terminal with a unique name so we can change the number of bits without getting an error
                    temp_new_term_name = get_unique_term_name(cvid new_term_name)
                    schCreateSymbolPin(cvid pin_master temp_new_term_name pin_direction pin_location "R0" )

                    new_term = dbFindTermByName(cvid temp_new_term_name )
                    dbDeleteObject(term)
                    new_term~>name = new_term_name
                else
                    ; remove pin
                    dbDeleteObject(pin~>fig)
                    dbDeleteObject(pin)
                    dbDeleteObject(term)
                )
            )
        )

        ; add new pins
        when( new_pins
            modified_pins = 't
            ; get location for new pins
            npin_xl = xCoord(car(pin_xy_info))
            npin_yl = yCoord(car(pin_xy_info)) - 2 * snap_dist
            npin_xr = xCoord(cadr(pin_xy_info))
            npin_yr = yCoord(cadr(pin_xy_info)) - 2 * snap_dist
            foreach( npin_info new_pins
                npin_name = car(npin_info)
                npin_type = cadr(npin_info)
                npin_stype = caddr(npin_info)

                ; verify that this pin does not exist yet
                when(dbFindTermByName(cvid npin_name)
                    dbClose(pin_master)
                    dbReopen(cvid, "r")
                    dbClose(cvid)
                    error( "Terminal %s already exists" npin_name)
                )

                ; update pin order
                new_port_order = tconc(new_port_order npin_name)

                ; get pin location based on pin type
                if( equal(npin_type "output") then
                    label_location = npin_xr:npin_yr
                    label_rel_location = "lowerLeft"
                    npin_yr = npin_yr - 2 * snap_dist
                else
                    label_location = npin_xl:npin_yl
                    label_rel_location = "lowerRight"
                    npin_yl = npin_yl - 2 * snap_dist
                )

                ; create label and pin
                schCreateSymbolLabel(cvid label_location "pin label" npin_name label_rel_location
                                     "R0" "stick" snap_dist "normalLabel")
                schCreateSymbolPin(cvid pin_master npin_name npin_type label_location "R0" 't nil nil npin_stype)
            )
        )

        dbClose(pin_master)

        when( modified_pins
            ; update pin order
            new_port_order = cdar(new_port_order)
            schEditPinOrder(cvid new_port_order 't)
            dbSave(cvid)

            ; update termOrder for each simulators
            cell_obj = ddGetObj(lib_name cell_name nil nil nil "r")
            unless( bc = cdfGetBaseCellCDF(cell_obj)
                ddReleaseObj(cell_obj)
                dbReopen(cvid, "r")
                dbClose(cvid)
                error("Cannot find CDF parameters for %s__%s.  Delete generated cell and try again" lib_name cell_name)
            )
            foreach( simu simulators
                get(bc->simInfo simu)->termOrder = new_port_order
            )
            unless( cdfSaveCDF(bc)
                ddReleaseObj(cell_obj)
                dbReopen(cvid, "r")
                dbClose(cvid)
                error("Cannot save termOrder CDF for %s__%s." lib_name cell_name)
            )
            ddReleaseObj(cell_obj)
        )
        ; opening schematic will open all symbols inside that schematic.
        ; as the result, dbClose may not close this symbol view.  To get rid
        ; of edit lock, we use dbReopen so even if dbClose fails the edit lock
        ; will be gone.
        dbReopen(cvid, "r")
        dbClose(cvid)
    )
)

; get all the wire objects connected to terminals of the given instance.
; we assume each terminal has exactly one pin with 1 wire connected, with a
; single label on the wire.  The wire doesn't connect to anything else.
; returns an association list from terminal name to a list of net name and wire figure object.
procedure( get_instance_terminal_wires(sch inst "gg")
    let( (snap_dist term_name pin_fig xval yval inst_term_xy net_name ans net_map)
        ans = nil
        net_map = nil
        snap_dist = schGetEnv("schSnapSpacing")
        foreach( inst_term inst~>instTerms
            term_name = inst_term~>name
            when( inst_term~>term~>pinCount != 1
                dbClose(sch)
                error("Terminal %s must have exactly one pin." term_name)
            )
            unless( pin_fig = car(inst_term~>term~>pins)~>fig
                dbClose(sch)
                error("Cannot find pin figure for terminal %s" term_name)
            )
            ; get instance terminal coordinate in schematic
            bbox = dbTransformBBox(pin_fig~>bBox inst~>transform)
            ; printf("terminal pin fig bbox: %A\n" bbox)
            xval = xCoord(car(bbox)) + (xCoord(cadr(bbox)) - xCoord(car(bbox))) / 2.0
            yval = yCoord(car(bbox)) + (yCoord(cadr(bbox)) - yCoord(car(bbox))) / 2.0
            ; quantize to schematic snap spacing to avoid floating point rounding error.
            inst_term_xy = round2(xval / snap_dist) * snap_dist:round2(yval / snap_dist) * snap_dist
            net_name = inst_term~>net~>name
            net_map = cons(list(term_name net_name) net_map)
            ; printf("terminal pin x/y: %A\n" inst_term_xy)
            ; There's a bug here where sometimes a valid net does not have figs. Solution: delete wires always.
            foreach( fig inst_term~>net~>figs
                points = fig~>points
                when( member(inst_term_xy points)
                    when( length(points) != 2
                        error("pin for terminal %s must be connected to a single wire with label" term_name)
                    )
                    ; printf("adding figure for terminal %s\n" term_name)
                    ans = cons(list(term_name fig) ans)
                )
            )
        )
        list(ans net_map)
    )
)

; record an association list from pin name to pin location in units of snap distances.
; the pin name is sorted alphabetically so we can use the equal function to test
; for equality.
procedure( get_instance_pin_info(inst "g")
    let( (snap_dist term_name pin_fig xval yval inst_term_xy ans)
        ans = nil
        snap_dist = schGetEnv("schSnapSpacing")
        foreach( term inst->master->terminals
            term_name = term~>name
            ; get terminal coordinate in symbol
            pin_fig = car(term~>pins)~>fig
            bbox = pin_fig~>bBox
            xval = (xCoord(car(bbox)) + xCoord(cadr(bbox))) / 2.0
            yval = (yCoord(car(bbox)) + yCoord(cadr(bbox))) / 2.0
            ; quantize to schematic snap spacing to avoid floating point rounding error.
            inst_term_xy = round2(xval / snap_dist):round2(yval / snap_dist)
            ans = cons(list(term_name inst_term_xy) ans)
        )
        sortcar(ans nil)
    )
)

; Modify the instance terminal connections of the given instance.
; we assume each terminal to modify has at most 1 wire connected,
; if it exists, the wire connects to nothing else, and it has a label.
; In this way, this function just have to change the label text.
;
; if wire_list is not empty, then that means each terminal has exactly one
; wire connected.  This function will update the label on the wires according
; to term_mapping.
;
; if wire_list is empty, then that means no wires are connected to terminals.
; this function will attach labels directly to each terminal.  The labels are
; determined first from term_mapping, then from net_map
;
; sch is the schematic database object.  Must be opened in append/write mode.
; inst is the instance object to modify.
; term_mapping is a list of key-value pairs, where keys are old net names,
; and values are new net names.
procedure( modify_instance_terminal(sch inst wire_list net_map term_mapping "gglll")
    let( (snap_dist key_val old_name new_name fig points mid_point new_wire inst_term inst_pin
          bbox xval yval term_map_final db_term)
        ; get schematic snap distance spacing.
        snap_dist = schGetEnv("schSnapSpacing")
        if( wire_list then
            foreach( wire_info wire_list
                old_name = car(wire_info)
                when(key_val = assoc(old_name term_mapping)
                    new_name = cadr(key_val)
                    fig = cadr(wire_info)
                    points = fig~>points
                    mid_point = foreach(mapcar (c1 c2) car(points) cadr(points) (c1 + c2) / 2.0)
                    ; delete old wire, then add wire back with new label.
                    schDelete(fig)
                    new_wire = car(schCreateWire(sch "draw" "full" points snap_dist snap_dist 0))
                    schCreateWireLabel(sch new_wire mid_point new_name "lowerCenter" "R0" "stick" 0.0625 nil)
                )
            )
            't
        else
            ; combine net_map and term_mapping
            term_map_final = copy(term_mapping)
            foreach( net_info net_map
                old_name = car(net_info)
                unless( assoc(old_name term_map_final)
                    ; add net mapping only if it's not in term_mapping
                    term_map_final = cons(net_info term_map_final)
                )
            )
            foreach( net_info term_map_final
                old_name = car(net_info)
                new_name = cadr(net_info)

                when(db_term = dbFindTermByName(inst->master old_name)
                    ; only create terminal that's present in the current master
                    inst_term = dbCreateInstTerm(nil inst db_term)
                    inst_pin = car(inst_term~>term~>pins)~>fig
                    bbox = dbTransformBBox(inst_pin~>bBox inst~>transform)
                    xval = (xCoord(car(bbox)) + xCoord(cadr(bbox))) / 2.0
                    yval = (yCoord(car(bbox)) + yCoord(cadr(bbox))) / 2.0
                    xval = round2(xval / snap_dist) * snap_dist
                    yval = round2(yval / snap_dist) * snap_dist

                    new_wire = car(schCreateWire(sch "draw" "full" list(xval-snap_dist:yval-snap_dist xval:yval)
                                   snap_dist snap_dist 0))
                    schCreateWireLabel(sch new_wire xval:yval new_name "lowerCenter" "R0" "stick" 0.0625 nil)
                )
            )
            't
        )
    )
)

; modify a schematic cell.  Used to convert copied template cells into concrete instantiation.
;
; inst_list is an association list of (inst_name, rinst_list) pairs.  Where:
;
; inst_name : name of the instance in the template cell.
; rinst_list : a list of rinsts, which are instances to replace the original instance by.
;              If this list is empty, the original instance should be deleted.  If the list
;              has more than one element, we should array the original instance.
;
; Each rinst is a disembodied property lists, with the properties:
;
; rinst->name : the name of this rinst.
; rinst->lib_name : the instance master library.
; rinst->cell_name : the instance master cell.
; rinst->params : an association list of the CDF params of this rinst.  The values are always string.
; rinst->term_mapping : an association list of the modified terminal connections of this rinst.
;                       if no connections are changed, this list should be empty.
;
; (You can read more about disembodied property lists and association list in the skill
; language user guide).
;
; For each instance, this function does the following:
; 1. Find the instance with the given name.
; 2. If rinst_list is nil, delete this instance.
; 3. If rinst_list has exactly one element:
;    i.   rename the instance name to rinst's name.
;    ii.  change the instance master of the instance.
;    iii. change the CDF parameters (this should only happen with BAG primitives).
;    iv.  change the port connections of this instance.
; 4. If rinst_list has more than one element, for each additional element,
;    copy the original instance and perform step 3 on that instance.
;
; This procedure allows one to delete or array any instances in the schematic template.
procedure( modify_schematic_content(sch_cv inst_list "gl")
    let( (inst_obj inst_name rinst_list rinst_len cur_inst wire_list net_map par_val xl yb xr yt transform
          snap_dist errmsg pin_info tmp_result xshift yshift xcoord ycoord dbu_per_uu)
        snap_dist = schGetEnv("schSnapSpacing")
        dbu_per_uu = sch_cv~>DBUPerUU
        foreach( inst inst_list
            inst_name = car(inst)
            unless( inst_obj = dbFindAnyInstByName(sch_cv inst_name)
                dbClose(sch_cv)
                error( "Cannot find instance %s" inst_name )
            )
            rinst_list = cadr(inst)
            rinst_len = length(rinst_list)
            last_inst = nil
            if( rinst_len == 0 then
                ; no instances to replace by, delete.
                wire_list = car(get_instance_terminal_wires(sch_cv inst_obj))
                ; delete wires connected to instance
                foreach( wire_info wire_list
                    schDelete(cadr(wire_info))
                )
                wire_list = nil
                ; delete instance
                dbDeleteObject(inst_obj)
            else
                tmp_result = get_instance_terminal_wires(sch_cv inst_obj)
                wire_list = car(tmp_result)
                net_map = cadr(tmp_result)
                foreach( wire_info wire_list
                    schDelete(cadr(wire_info))
                )
                wire_list = nil
                xl = xCoord(car(inst_obj~>bBox))
                yb = yCoord(car(inst_obj~>bBox))
                xr = xCoord(cadr(inst_obj~>bBox))
                yt = yCoord(cadr(inst_obj~>bBox))

                foreach( rinst rinst_list
                    if(rinst->dx == 0 then
                        xcoord = 0
                    else
                        xcoord = round2((rinst->dx / dbu_per_uu) / snap_dist) * snap_dist
                    )
                    if(rinst->dy == 0 then
                        ycoord = 0
                    else
                        ycoord = round2((rinst->dy / dbu_per_uu) / snap_dist) * snap_dist
                    )

                    transform = list(xcoord:ycoord "R0" 1.0)
                    ; printf("instance %s transform: %A\n" cur_inst~>name transform)

                    ; copy instance
                    if(xcoord == 0 && ycoord == 0 then
                        cur_inst = inst_obj
                    else
                        cur_inst = dbCopyFig(inst_obj nil transform)
                        tmp_result = get_instance_terminal_wires(sch_cv cur_inst)
                        net_map = cadr(tmp_result)
                    )
                    ; change instance name and master
                    when(cur_inst->name != rinst->name
                        cur_inst->name = rinst->name
                    )
                    schReplaceProperty(list(cur_inst) "master" sprintf(nil "%s %s %s" rinst->lib_name
                                                                       rinst->cell_name cur_inst->viewName))
                    ; set parameters
                    foreach( cdf_par cdfGetInstCDF(cur_inst)~>parameters
                        par_val = cadr(assoc(cdf_par->name rinst->params))
                        ; change CDF parameter value only if specified in given parameters
                        when( par_val != nil
                            cdf_par->value = par_val
                        )
                    )

                    ; modify connections
                    modify_instance_terminal(sch_cv cur_inst wire_list net_map rinst->term_mapping)
                )
            )
        )
    )
)

; update pins of a schematic
; cvid is the opened cellview id of the schematic.  It must be in append mode.
; pin_map is a list of two-element lists of old pin names and new pin names, respectively.
; ipin, opin, and iopin are lists of three strings for input/output/inout pins, respectively.
; first element is the pin master library, second element is the pin mater cell, and third element
; is the pin master cellview.
procedure( update_schematic_pin(cvid pin_map new_pins ipin opin iopin "glllll")
    let( (snap_dist cur_term_name new_term_name term pin pin_orient pin_location pin_direction
          temp_new_term_name pin_master ipin_master opin_master iopin_master
          pin_xy_info npin_xl npin_yl npin_xr npin_yr npin_name npin_type)

        snap_dist = schGetEnv("schSnapSpacing")

        ; open pin masters
        ipin_master = open_pin_master(cvid ipin)
        opin_master = open_pin_master(cvid opin)
        iopin_master = open_pin_master(cvid iopin)
        pin_master = nil

        ; get new pin locations before any pin addition/substraction.
        pin_xy_info = get_new_pin_locations(cvid snap_dist)

        ; rename or remove pins
        foreach( p pin_map
            cur_term_name = car(p)
            new_term_name = cadr(p)
            ; printf("%s %s\n" cur_term_name new_term_name)
            when(cur_term_name != new_term_name
                unless( term = dbFindTermByName(cvid cur_term_name)
                    dbClose(cvid)
                    dbClose(ipin_master)
                    dbClose(opin_master)
                    dbClose(iopin_master)
                    error( "Terminal %s not found." cur_term_name )
                )
                when( term~>pinCount != 1
                    dbClose(cvid)
                    dbClose(ipin_master)
                    dbClose(opin_master)
                    dbClose(iopin_master)
                    error( "Terminal %s does not have exactly one pin." cur_term_name)
                )
                pin = car(term~>pins)

                if( strlen(new_term_name) != 0 then
                    ; rename pin
                    pin_orient = pin~>fig~>orient
                    pin_location = pin~>fig~>xy
                    pin_direction = term~>direction

                    ; create new pin figure
                    cond( ( pin_direction == "input" pin_master = ipin_master)
                          ( pin_direction == "output" pin_master = opin_master)
                          ( 't pin_master = iopin_master)
                    )

                    ; delete pin
                    unless( dbDeleteObject(pin~>fig)
                        dbClose(cvid)
                        dbClose(ipin_master)
                        dbClose(opin_master)
                        dbClose(iopin_master)
                        error( "Cannot delete pin for terminal %s" cur_term_name )
                    )

                    ; create a temporary terminal with a unique name so we can change the number of bits without getting an error
                    temp_new_term_name = get_unique_term_name(cvid new_term_name)
                    schCreatePin(cvid pin_master temp_new_term_name pin_direction nil pin_location "R0" )

                    ; now rename the new terminal
                    new_term = dbFindTermByName(cvid temp_new_term_name )
                    new_term~>name = new_term_name
                else
                    ; remove pin
                    dbDeleteObject(pin~>fig)
                )
            )
        )

        ; add new pins
        when( new_pins
            ; get location for new pins
            npin_xl = xCoord(car(pin_xy_info)) - 2 * snap_dist
            npin_yl = yCoord(car(pin_xy_info)) - 2 * snap_dist
            npin_xr = xCoord(cadr(pin_xy_info))
            npin_yr = yCoord(cadr(pin_xy_info)) - 2 * snap_dist
            foreach( npin_info new_pins
                npin_name = car(npin_info)
                npin_type = cadr(npin_info)
                npin_stype = caddr(npin_info)

                ; verify that this pin does not exist yet
                when(dbFindTermByName(cvid npin_name)
                    dbClose(cvid)
                    dbClose(ipin_master)
                    dbClose(opin_master)
                    dbClose(iopin_master)
                    error( "Terminal %s already exists" npin_name)
                )

                ; get pin location based on pin type
                cond( ( npin_type == "input" pin_master = ipin_master pin_location = npin_xl:npin_yl npin_yl = npin_yl - 2 * snap_dist)
                      ( npin_type == "output" pin_master = opin_master pin_location = npin_xr:npin_yr npin_yr = npin_yr - 2 * snap_dist)
                      ( 't pin_master = iopin_master pin_location = npin_xl:npin_yl npin_yl = npin_yl - 2 * snap_dist)
                )
                ; create pin
                schCreatePin(cvid pin_master npin_name npin_type nil pin_location "R0" nil nil npin_stype)
            )
        )

        dbClose(ipin_master)
        dbClose(opin_master)
        dbClose(iopin_master)
    )
)

; Perform check-and-save on the given schematic database object, then close it.
procedure( check_and_save_schematic(sch "g")
    let( (errs)
        schSetEnv( "checkHierSave" 't)
        schSetEnv( "saveAction" "Save")
        errs = schCheckHier(sch "schematic symbol" "")
        foreach( ex errs
            warn( "%s__%s (%s) has %d errors." car(ex)~>lib~>name car(ex)~>cellName car(ex)~>viewName cadr(ex))
        )
        ; make sure all edit locks are gone by reopening in read mode
        dbReopen(sch, "r")
        dbClose(sch)
    )
)

; given a copied template cell, modify it to a concrete schematic.
procedure( convert_template_cells(lib_name cell_name pin_map new_pins inst_list sympin ipin opin iopin simulators)
    let( (sym_cv sch)
        ; update symbol view first.
        if( sym_cv = dbOpenCellViewByType(lib_name cell_name "symbol" nil "r") then
            printf("*INFO* Updating %s__%s symbol pins.\n" lib_name cell_name)
            update_symbol_pin(lib_name cell_name pin_map new_pins sympin simulators)
        else
            warn("Did not find symbol for %s__%s.  Skipping.  Is it testbench?" lib_name cell_name)
        )

        ; attempt to open schematic in append mode
        unless( sch = dbOpenCellViewByType(lib_name cell_name "schematic" nil "a")
            error("Cannot open %s__%s (schematic) in append mode." lib_name cell_name)
        )
        ; update schematic content
        printf("*INFO* Updating %s__%s instances and connections.\n" lib_name cell_name)
        modify_schematic_content(sch inst_list)
        ; update schematic pins
        printf("*INFO* Updating %s__%s schematic pins.\n" lib_name cell_name)
        update_schematic_pin(sch pin_map new_pins ipin opin iopin)
        check_and_save_schematic(sch)
    )
)

; create concrete schematics
procedure( create_concrete_schematic( lib_name tech_lib lib_path temp_file change_file
                                      sympin ipin opin iopin simulators copy "tttttlllllg" )
    let( (template_list change_list cell_name pin_map inst_list)
        printf("*INFO* Reading template and change list from file\n")
        template_list = parse_data_from_file( temp_file )
        change_list = parse_data_from_file( change_file )
        when( copy
            printf("*INFO* Creating library: %s\n" lib_name)
            create_or_erase_library( lib_name tech_lib lib_path nil )
            printf("*INFO* Copying templates to library: %s\n" lib_name)
            copy_templates_to_library( lib_name template_list )
        )
        foreach( change change_list
            cell_name = change->name
            pin_map = change->pin_map
            new_pins = change->new_pins
            inst_list = change->inst_list
            printf("*INFO* Updating cell %s__%s\n" lib_name cell_name)
            convert_template_cells( lib_name cell_name pin_map new_pins inst_list
                                    sympin ipin opin iopin simulators )
        )
        't
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  BAG server related functions            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure( stdoutHandler(ipcId data)
    let( (result result_str)
        if( bag_server_started > 0 then
            printf("*INFO* Evaluate expression from BAG process: %s\n" data)
            if( result = errsetstring(data 't) then
                sprintf(result_str "%A\n" car(result))
            else
                sprintf(result_str "%s\n" car(nthelem(5 errset.errset)))
            )
            printf("*INFO* Sending result to BAG process: %s" result_str)
            ipcWriteProcess(ipcId sprintf(nil "%d\n" strlen(result_str)))
            ipcWriteProcess(ipcId result_str)
            't
        else
            if( data == "BAG skill server has started.  Yay!\n" then
                bag_server_started = 1
                printf("*INFO* BAG skill server started.\n")
            else
                printf("*INFO* Waiting for BAG skill server.  Message: %s\n" data)
            )
        )
    )
)

procedure( stderrHandler(ipcId data)
    warn("BAG server process error: %s\n" data)
    warn("Shutting down BAG server.")
    ipcKillProcess(ipcId)
    't
)

procedure( exitHandler(ipcId exitId)
    printf("*INFO* BAG server process exited with status: %d\n" exitId)
    't
)

procedure( start_bag_server()
    bag_server_started = 0
    printf("*INFO* Starting BAG server process.\n")
    ipcBeginProcess("bash virt_server.sh" "" 'stdoutHandler 'stderrHandler 'exitHandler "")
)

bag_server_started = 0
bag_modify_adexl_counter = 0
bag_proc = start_bag_server()
